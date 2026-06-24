#include "engine/runtime.h"

#include "common/logger.h"
#include "engine/custom_compute_service.h"
#include "engine/runtime_internal.h"
#include "engine/shared_buffer_service.h"

namespace cressim::neo::engine
{

namespace
{

bool hasGraphicsBackendContext(gpu::GpuDevice &device)
{
    gpu::GpuGraphicsBackendContext graphicsContext{};
    return device.tryGetGraphicsBackendContext(graphicsContext) &&
           graphicsContext.renderDevice != nullptr && graphicsContext.graphicsContext != nullptr;
}

bool hasPhysicsBackendContext(gpu::GpuDevice &device)
{
    gpu::GpuComputeBackendContext computeContext{};
    return device.tryGetPhysicsBackendContext(computeContext) &&
           computeContext.renderDevice != nullptr && computeContext.computeContext != nullptr;
}

void ensureDeviceFrameActive(gpu::GpuDevice *device, const common::FrameContext &frameContext,
                             bool &inOutDeviceFrameActive)
{
    if (device == nullptr || inOutDeviceFrameActive)
    {
        return;
    }

    device->beginFrame(frameContext);
    inOutDeviceFrameActive = true;
}

bool syncGpuScene(World &world, gpu::GpuDevice *device, RenderSceneUploader *uploader,
                  physics::PhysicsSolver *physicsSolver, bool usePhysicsPoses)
{
    if (uploader == nullptr)
    {
        world.setGpuEntityScene({});
        return false;
    }

    bool gpuSceneReady = uploader->uploadEntityPoseData(world.renderObjectPositions(),
                                                        world.renderObjectOrientations(),
                                                        world.renderObjectScales());
    if (gpuSceneReady && usePhysicsPoses && physicsSolver != nullptr)
    {
        if (!uploader->applyMappedEntityPoses(physicsSolver->gpuSceneView().rigid.poses,
                                              world.physicsRenderableMappings()))
        {
            gpuSceneReady = false;
        }
    }

    if (gpuSceneReady && uploader->uploadRenderableMetadata(world.renderableMetadata()) &&
        uploader->uploadRenderableQueueInfo(world.renderableQueueInfo()) &&
        uploader->uploadSoftBodyVertexBindings(world.softBodyVertexBindings()) &&
        uploader->uploadCameraInputs(world.cameraInputs()) &&
        uploader->uploadLightInputs(world.lightInputs()) &&
        uploader->uploadLocalLightSelections(world.localLightSelections()) &&
        (!device || device->waitForPhysicsOnGraphics()))
    {
        world.setGpuEntityScene(uploader->sceneView());
        return true;
    }

    CRESSIM_LOG_WARNING("Runtime: GPU scene sync failed.");
    world.setGpuEntityScene({});
    return false;
}

} // namespace

Runtime::Runtime() = default;

Runtime::~Runtime()
{
    shutdown();
}

bool Runtime::initialize(const RuntimeConfig &config)
{
    if (mInitialized)
    {
        return true;
    }

    mGpuDevice = gpu::createGpuDevice();
    if (!mGpuDevice)
    {
        return false;
    }

    if (!mGpuDevice->initialize(config.gpuDeviceDesc))
    {
        mGpuDevice.reset();
        return false;
    }

    mPhysicsSolver = std::make_unique<physics::PhysicsSolver>(*mGpuDevice, config.physicsDesc);
    if (!mPhysicsSolver || !mPhysicsSolver->initialize())
    {
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mUltrasoundSystem = std::make_unique<UltrasoundSystem>(*mGpuDevice, *mPhysicsSolver);
    if (!mUltrasoundSystem || !mUltrasoundSystem->initialize())
    {
        mUltrasoundSystem.reset();
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mWorld.setSceneLayout(config.sceneLayout);

    if (hasGraphicsBackendContext(*mGpuDevice) && hasPhysicsBackendContext(*mGpuDevice))
    {
        mRenderSceneUploader = std::make_unique<RenderSceneUploader>(*mGpuDevice);
        if (!mRenderSceneUploader || !mRenderSceneUploader->initialize(config.sceneLayout))
        {
            mRenderSceneUploader.reset();
            mUltrasoundSystem->shutdown();
            mUltrasoundSystem.reset();
            mPhysicsSolver->shutdown();
            mPhysicsSolver.reset();
            mGpuDevice->shutdown();
            mGpuDevice.reset();
            return false;
        }
    }

    mRenderer = std::make_unique<graphics::Renderer>(*mGpuDevice, mResources, config.rendererDesc);
    if (!mRenderer->initialize())
    {
        mRenderer.reset();
        if (mRenderSceneUploader)
        {
            mRenderSceneUploader->shutdown();
            mRenderSceneUploader.reset();
        }
        if (mUltrasoundSystem)
        {
            mUltrasoundSystem->shutdown();
            mUltrasoundSystem.reset();
        }
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mCustomComputeService = std::make_unique<CustomComputeService>(*mGpuDevice);
    mSharedBufferService  = std::make_unique<SharedBufferService>(*mGpuDevice);
    mInitialized          = true;
    return true;
}

void Runtime::shutdown()
{
    if (!mInitialized)
    {
        return;
    }

    if (mDeviceFrameActive && mGpuDevice)
    {
        mGpuDevice->endFrame(mLastFrameContext);
        mDeviceFrameActive = false;
    }

    mRenderer.reset();
    if (mCustomComputeService)
    {
        mCustomComputeService->clear();
        mCustomComputeService.reset();
    }
    if (mSharedBufferService)
    {
        mSharedBufferService->clear();
        mSharedBufferService.reset();
    }

    if (mRenderSceneUploader)
    {
        mRenderSceneUploader->shutdown();
        mRenderSceneUploader.reset();
    }
    if (mUltrasoundSystem)
    {
        mUltrasoundSystem->shutdown();
        mUltrasoundSystem.reset();
    }

    if (mPhysicsSolver)
    {
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
    }

    if (mGpuDevice)
    {
        mGpuDevice->shutdown();
        mGpuDevice.reset();
    }

    mLastRenderStats    = {};
    mRenderFrameOptions = {};
    mLastFrameContext   = {};
    mDeviceFrameActive  = false;
    mWorldUploaded      = false;
    mHasPhysicsState    = false;
    mInitialized        = false;
}

void Runtime::prepare()
{
    if (!mInitialized)
    {
        return;
    }

    mWorld.ensureRenderStateUpToDate(mResources);
    if (mUltrasoundSystem)
    {
        if (!mUltrasoundSystem->prepare(mWorld))
        {
            CRESSIM_LOG_WARNING("Runtime: ultrasound prepare failed.");
        }
    }
    mWorldUploaded   = false;
    mHasPhysicsState = false;
}

bool Runtime::uploadWorld()
{
    if (!mInitialized)
    {
        return false;
    }

    bool uploaded = true;
    if (mPhysicsSolver && !mPhysicsSolver->syncWorldState(mWorld.physicsWorld()))
    {
        CRESSIM_LOG_ERROR("Runtime: physics world upload failed.");
        uploaded = false;
    }

    if (uploaded && mRenderSceneUploader &&
        !syncGpuScene(mWorld, mGpuDevice.get(), mRenderSceneUploader.get(), mPhysicsSolver.get(),
                      false))
    {
        uploaded = false;
    }

    mWorldUploaded   = uploaded;
    mHasPhysicsState = false;
    return uploaded;
}

bool Runtime::stepPhysics(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return false;
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: stepPhysics() requires uploadWorld() after prepare() and "
                          "before execution.");
        return false;
    }

    mLastFrameContext = frameContext;

    bool physicsStepSucceeded = true;
    if (mPhysicsSolver)
    {
        physicsStepSucceeded = mPhysicsSolver->step(frameContext, mWorld.physicsWorld());
        if (!physicsStepSucceeded)
        {
            CRESSIM_LOG_ERROR("Runtime: physics step failed at frame ", frameContext.frameIndex,
                              " (dt=", frameContext.deltaSeconds, ").");
        }
    }
    if (physicsStepSucceeded)
    {
        mHasPhysicsState = true;
    }
    return physicsStepSucceeded;
}

bool Runtime::stepSimulationSensors(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return false;
    }

    mLastFrameContext = frameContext;

    if (!mUltrasoundSystem)
    {
        return true;
    }

    ensureDeviceFrameActive(mGpuDevice.get(), frameContext, mDeviceFrameActive);

    const bool succeeded = mUltrasoundSystem->execute(frameContext, mWorld);
    if (!succeeded)
    {
        CRESSIM_LOG_WARNING("Runtime: ultrasound step failed at frame ", frameContext.frameIndex,
                            ".");
    }
    return succeeded;
}

void Runtime::stepVisualSensors(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    (void)syncGpuScene(mWorld, mGpuDevice.get(), mRenderSceneUploader.get(), mPhysicsSolver.get(),
                       mHasPhysicsState);

    mLastFrameContext = frameContext;

    ensureDeviceFrameActive(mGpuDevice.get(), frameContext, mDeviceFrameActive);

    physics::PhysicsGpuSceneView physicsSceneView{};
    const physics::PhysicsGpuSceneView *physicsScenePtr = nullptr;
    if (mPhysicsSolver)
    {
        physicsSceneView = mPhysicsSolver->gpuSceneView();
        physicsScenePtr  = &physicsSceneView;
    }

    mLastRenderStats = mRenderer->render(frameContext, mWorld.hostSceneView(), physicsScenePtr,
                                         mRenderFrameOptions);
}

void Runtime::endFrame(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    mLastFrameContext = frameContext;
    if (!mDeviceFrameActive || !mGpuDevice)
    {
        return;
    }

    mGpuDevice->endFrame(frameContext);
    mDeviceFrameActive = false;
}

World &Runtime::getWorld() noexcept
{
    return mWorld;
}

const World &Runtime::getWorld() const noexcept
{
    return mWorld;
}

gpu::GpuDevice *Runtime::getGpuDevice() noexcept
{
    return mGpuDevice.get();
}

const gpu::GpuDevice *Runtime::getGpuDevice() const noexcept
{
    return mGpuDevice.get();
}

physics::PhysicsSolver *Runtime::getPhysicsSolver() noexcept
{
    return mPhysicsSolver.get();
}

const physics::PhysicsSolver *Runtime::getPhysicsSolver() const noexcept
{
    return mPhysicsSolver.get();
}

const graphics::RenderStats &Runtime::lastRenderStats() const noexcept
{
    return mLastRenderStats;
}

void Runtime::setRenderFrameOptions(const graphics::RenderFrameOptions &options) noexcept
{
    mRenderFrameOptions = options;
}

const graphics::RenderFrameOptions &Runtime::renderFrameOptions() const noexcept
{
    return mRenderFrameOptions;
}

graphics::RenderResourceManager &Runtime::getResources() noexcept
{
    return mResources;
}

const graphics::RenderResourceManager &Runtime::getResources() const noexcept
{
    return mResources;
}

SharedBufferHandle Runtime::createSharedBuffer(const SharedBufferDesc &desc)
{
    return mInitialized && mSharedBufferService != nullptr
               ? mSharedBufferService->createBuffer(desc)
               : SharedBufferHandle{};
}

bool Runtime::destroySharedBuffer(const SharedBufferHandle handle)
{
    return mInitialized && mSharedBufferService != nullptr &&
           mSharedBufferService->destroyBuffer(handle);
}

std::vector<SharedBufferInfo> Runtime::listSharedBuffers() const
{
    return mInitialized && mSharedBufferService != nullptr ? mSharedBufferService->listBuffers()
                                                           : std::vector<SharedBufferInfo>{};
}

bool Runtime::tryGetSharedBufferInfo(const SharedBufferHandle handle,
                                     SharedBufferInfo &outInfo) const
{
    return mInitialized && mSharedBufferService != nullptr &&
           mSharedBufferService->tryGetBufferInfo(handle, outInfo);
}

bool Runtime::tryGetSharedBufferCudaView(const SharedBufferHandle handle,
                                         SharedBufferCudaView &outView) const
{
    return mInitialized && mSharedBufferService != nullptr &&
           mSharedBufferService->tryGetCudaView(handle, outView);
}

std::shared_ptr<void> Runtime::retainSharedBuffer(const SharedBufferHandle handle) const
{
    return mInitialized && mSharedBufferService != nullptr
               ? mSharedBufferService->retainBuffer(handle)
               : std::shared_ptr<void>{};
}

std::shared_ptr<void> RuntimeInternalAccess::retainSharedBufferLease(
    const Runtime &runtime, const SharedBufferHandle handle)
{
    return runtime.retainSharedBuffer(handle);
}

bool Runtime::syncSharedBufferToCuda(const SharedBufferHandle handle)
{
    if (!mInitialized || mSharedBufferService == nullptr || mGpuDevice == nullptr)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mGpuDevice->tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        return false;
    }

    return mSharedBufferService->synchronizeToCuda(handle, computeBackend.computeContext);
}

bool Runtime::syncSharedBufferFromCuda(const SharedBufferHandle handle)
{
    if (!mInitialized || mSharedBufferService == nullptr || mGpuDevice == nullptr)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mGpuDevice->tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        return false;
    }

    return mSharedBufferService->synchronizeFromCuda(handle, computeBackend.computeContext);
}

bool Runtime::tryGetPreparedRigidLayoutMapping(RigidLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mInitialized || mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld    = mWorld.physicsWorld();
    const physics::RigidBodySoAHost &rigidBodies = physicsWorld.rigidBodySoA();
    const physics::ColliderSoAHost &colliders    = physicsWorld.colliderSoA();
    const physics::BodyColliderMappingHost &bodyColliderMapping =
        physicsWorld.bodyColliderMapping();
    outMapping.rigidBodyCount              = static_cast<std::uint32_t>(rigidBodies.size());
    outMapping.colliderCount               = static_cast<std::uint32_t>(colliders.size());
    outMapping.bindingGeneration           = physicsWorld.rigidBodyTopologyRevision();
    outMapping.rigidBodyEntityIds          = rigidBodies.entityIds;
    outMapping.rigidBodyEnvironmentIndices = rigidBodies.environmentIndices;
    outMapping.colliderIds                 = colliders.colliderIds;
    outMapping.colliderEntityIds           = colliders.entityIds;
    outMapping.colliderOwnerBodyIndices    = colliders.ownerRigidBodyIndices;
    outMapping.colliderEnvironmentIndices  = colliders.environmentIndices;
    outMapping.bodyColliderOffsets         = bodyColliderMapping.colliderOffsets;
    outMapping.bodyColliderCounts          = bodyColliderMapping.colliderCounts;
    outMapping.bodyColliderIndices         = bodyColliderMapping.colliderIndices;
    return true;
}

std::vector<CustomComputeResourceDesc> Runtime::listCustomComputeResources()
{
    if (!mInitialized || mCustomComputeService == nullptr || mPhysicsSolver == nullptr)
    {
        return {};
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: listCustomComputeResources() requires uploadWorld() after "
                          "prepare() and before execution.");
        return {};
    }
    return mCustomComputeService->listResources(*mPhysicsSolver, mWorld.physicsWorld());
}

CustomComputePassHandle Runtime::createCustomComputePass(const CustomComputePassDesc &desc)
{
    if (!mInitialized || mCustomComputeService == nullptr || mPhysicsSolver == nullptr)
    {
        return {};
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: createCustomComputePass() requires uploadWorld() after "
                          "prepare() and before execution.");
        return {};
    }
    return mCustomComputeService->createPass(*mPhysicsSolver, mWorld.physicsWorld(),
                                             mSharedBufferService.get(), desc);
}

bool Runtime::updateCustomComputePassConstants(CustomComputePassHandle handle,
                                               const std::vector<std::uint8_t> &data)
{
    return mInitialized && mCustomComputeService != nullptr &&
           mCustomComputeService->updatePassConstants(handle, data);
}

bool Runtime::executeCustomComputePass(CustomComputePassHandle handle)
{
    if (!mInitialized || mCustomComputeService == nullptr || mPhysicsSolver == nullptr)
    {
        return false;
    }
    if (!mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: executeCustomComputePass() requires uploadWorld() after "
                          "prepare() and before execution.");
        return false;
    }
    return mCustomComputeService->executePass(*mPhysicsSolver, mWorld.physicsWorld(),
                                              mSharedBufferService.get(), handle);
}

bool Runtime::destroyCustomComputePass(CustomComputePassHandle handle)
{
    return mInitialized && mCustomComputeService != nullptr &&
           mCustomComputeService->destroyPass(handle);
}

} // namespace cressim::neo::engine
