#include "engine/runtime.h"

#include "common/logger.h"
#include "engine/custom_compute_service.h"
#include "engine/entity_scene_gpu_state.h"
#include "engine/render_scene_uploader.h"
#include "engine/shared_buffer_service.h"
#include "engine/ultrasound_system.h"
#include "engine/world.h"
#include "gpu/cuda_interop.h"
#include "version.h"

#include <algorithm>
#include <unordered_map>

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

bool syncGpuScene(World &world, EntitySceneGpuState *entitySceneState,
                  RenderSceneUploader *uploader, physics::PhysicsSolver *physicsSolver,
                  bool usePhysicsPoses, std::uint64_t &lastEntityPoseRevision,
                  std::uint64_t &lastRenderableMetadataRevision,
                  std::uint64_t &lastRenderableQueueInfoRevision,
                  std::uint64_t &lastSoftBodyVertexBindingRevision,
                  std::uint64_t &lastCameraInputRevision, std::uint64_t &lastLightInputRevision,
                  std::uint64_t &lastLocalLightSelectionRevision)
{
    if (entitySceneState == nullptr || uploader == nullptr)
    {
        world.setGpuEntityScene({});
        return false;
    }

    const std::uint64_t entityPoseRevision            = world.entityPoseRevision();
    const std::uint64_t renderableMetadataRevision    = world.renderableMetadataRevision();
    const std::uint64_t renderableQueueInfoRevision   = world.renderableQueueInfoRevision();
    const std::uint64_t softBodyVertexBindingRevision = world.softBodyVertexBindingRevision();
    const std::uint64_t cameraInputRevision           = world.cameraInputRevision();
    const std::uint64_t lightInputRevision            = world.lightInputRevision();
    const std::uint64_t localLightSelectionRevision   = world.localLightSelectionRevision();

    const bool needsEntityPoseUpload = entityPoseRevision != lastEntityPoseRevision;
    const bool needsRenderableMetadataUpload =
        renderableMetadataRevision != lastRenderableMetadataRevision;
    const bool needsRenderableQueueInfoUpload =
        renderableQueueInfoRevision != lastRenderableQueueInfoRevision;
    const bool needsSoftBodyVertexBindingUpload =
        softBodyVertexBindingRevision != lastSoftBodyVertexBindingRevision;
    const bool needsCameraInputUpload = cameraInputRevision != lastCameraInputRevision;
    const bool needsLightInputUpload  = lightInputRevision != lastLightInputRevision;
    const bool needsLocalLightSelectionUpload =
        localLightSelectionRevision != lastLocalLightSelectionRevision;

    bool gpuSceneReady = true;
    if (needsEntityPoseUpload)
    {
        gpuSceneReady = entitySceneState->uploadAuthoredEntityPoses(
            world.entityPosePositions(), world.entityPoseOrientations(), world.entityPoseScales());
    }
    if (gpuSceneReady && usePhysicsPoses && physicsSolver != nullptr)
    {
        if (!entitySceneState->applyMappedEntityPoses(physicsSolver->gpuSceneView().rigid.poses,
                                                      world.physicsRenderableMappings()))
        {
            gpuSceneReady = false;
        }
    }

    if (gpuSceneReady && needsRenderableMetadataUpload)
    {
        gpuSceneReady = uploader->uploadRenderableMetadata(world.renderableMetadata());
    }
    if (gpuSceneReady && needsRenderableQueueInfoUpload)
    {
        gpuSceneReady = uploader->uploadRenderableQueueInfo(world.renderableQueueInfo());
    }
    if (gpuSceneReady && needsSoftBodyVertexBindingUpload)
    {
        gpuSceneReady = uploader->uploadSoftBodyVertexBindings(world.softBodyVertexBindings());
    }
    if (gpuSceneReady && needsCameraInputUpload)
    {
        gpuSceneReady = uploader->uploadCameraInputs(world.cameraInputs());
    }
    if (gpuSceneReady && needsLightInputUpload)
    {
        gpuSceneReady = uploader->uploadLightInputs(world.lightInputs());
    }
    if (gpuSceneReady && needsLocalLightSelectionUpload)
    {
        gpuSceneReady = uploader->uploadLocalLightSelections(world.localLightSelections());
    }

    if (gpuSceneReady)
    {
        world.setGpuEntityScene(
            uploader->sceneView(entitySceneState->poseView(), entitySceneState->entityCount()));
        lastEntityPoseRevision            = entityPoseRevision;
        lastRenderableMetadataRevision    = renderableMetadataRevision;
        lastRenderableQueueInfoRevision   = renderableQueueInfoRevision;
        lastSoftBodyVertexBindingRevision = softBodyVertexBindingRevision;
        lastCameraInputRevision           = cameraInputRevision;
        lastLightInputRevision            = lightInputRevision;
        lastLocalLightSelectionRevision   = localLightSelectionRevision;
        return true;
    }

    CRESSIM_LOG_WARNING("Runtime: GPU scene sync failed.");
    world.setGpuEntityScene({});
    return false;
}

} // namespace

struct Runtime::Impl
{
    bool mInitialized = false;
    std::unique_ptr<gpu::GpuDevice> mGpuDevice;
    std::unique_ptr<EntitySceneGpuState> mEntitySceneGpuState;
    std::unique_ptr<RenderSceneUploader> mRenderSceneUploader;
    std::unique_ptr<physics::PhysicsSolver> mPhysicsSolver;
    std::unique_ptr<UltrasoundSystem> mUltrasoundSystem;
    std::unique_ptr<graphics::Renderer> mRenderer;
    std::unique_ptr<CustomComputeService> mCustomComputeService;
    std::unique_ptr<SharedBufferService> mSharedBufferService;
    graphics::RenderFrameOptions mRenderFrameOptions{};
    graphics::RenderStats mLastRenderStats{};
    World mWorld;
    graphics::RenderResourceManager mResources;
    common::FrameContext mLastFrameContext{};
    bool mDeviceFrameActive                                  = false;
    bool mWorldUploaded                                      = false;
    bool mPhysicsPosesNeedSync                               = false;
    std::uint64_t mLastUploadedEntityPoseRevision            = 0u;
    std::uint64_t mLastUploadedRenderableMetadataRevision    = 0u;
    std::uint64_t mLastUploadedRenderableQueueInfoRevision   = 0u;
    std::uint64_t mLastUploadedSoftBodyVertexBindingRevision = 0u;
    std::uint64_t mLastUploadedCameraInputRevision           = 0u;
    std::uint64_t mLastUploadedLightInputRevision            = 0u;
    std::uint64_t mLastUploadedLocalLightSelectionRevision   = 0u;
};

Runtime::Runtime() : mImpl(std::make_unique<Impl>()) {}

Runtime::~Runtime()
{
    shutdown();
}

bool Runtime::initialize(const RuntimeConfig &config)
{
    if (mImpl->mInitialized)
    {
        return true;
    }

    mImpl->mGpuDevice = gpu::createGpuDevice();
    if (!mImpl->mGpuDevice)
    {
        return false;
    }

    if (!mImpl->mGpuDevice->initialize(config.gpuDeviceDesc))
    {
        mImpl->mGpuDevice.reset();
        return false;
    }

    mImpl->mPhysicsSolver =
        std::make_unique<physics::PhysicsSolver>(*mImpl->mGpuDevice, config.physicsDesc);
    if (!mImpl->mPhysicsSolver || !mImpl->mPhysicsSolver->initialize())
    {
        mImpl->mPhysicsSolver.reset();
        mImpl->mGpuDevice->shutdown();
        mImpl->mGpuDevice.reset();
        return false;
    }

    mImpl->mUltrasoundSystem =
        std::make_unique<UltrasoundSystem>(*mImpl->mGpuDevice, *mImpl->mPhysicsSolver);
    if (!mImpl->mUltrasoundSystem || !mImpl->mUltrasoundSystem->initialize())
    {
        mImpl->mUltrasoundSystem.reset();
        mImpl->mPhysicsSolver->shutdown();
        mImpl->mPhysicsSolver.reset();
        mImpl->mGpuDevice->shutdown();
        mImpl->mGpuDevice.reset();
        return false;
    }

    mImpl->mWorld.setSceneLayout(config.sceneLayout);

    if (hasGraphicsBackendContext(*mImpl->mGpuDevice) &&
        hasPhysicsBackendContext(*mImpl->mGpuDevice))
    {
        mImpl->mEntitySceneGpuState = std::make_unique<EntitySceneGpuState>(*mImpl->mGpuDevice);
        mImpl->mRenderSceneUploader = std::make_unique<RenderSceneUploader>(*mImpl->mGpuDevice);
        if (!mImpl->mEntitySceneGpuState || !mImpl->mEntitySceneGpuState->initialize() ||
            !mImpl->mRenderSceneUploader ||
            !mImpl->mRenderSceneUploader->initialize(config.sceneLayout))
        {
            if (mImpl->mEntitySceneGpuState)
            {
                mImpl->mEntitySceneGpuState->shutdown();
                mImpl->mEntitySceneGpuState.reset();
            }
            mImpl->mRenderSceneUploader.reset();
            mImpl->mUltrasoundSystem->shutdown();
            mImpl->mUltrasoundSystem.reset();
            mImpl->mPhysicsSolver->shutdown();
            mImpl->mPhysicsSolver.reset();
            mImpl->mGpuDevice->shutdown();
            mImpl->mGpuDevice.reset();
            return false;
        }
    }

    mImpl->mRenderer = std::make_unique<graphics::Renderer>(*mImpl->mGpuDevice, mImpl->mResources,
                                                            config.rendererDesc);
    if (!mImpl->mRenderer->initialize())
    {
        mImpl->mRenderer.reset();
        if (mImpl->mRenderSceneUploader)
        {
            mImpl->mRenderSceneUploader->shutdown();
            mImpl->mRenderSceneUploader.reset();
        }
        if (mImpl->mEntitySceneGpuState)
        {
            mImpl->mEntitySceneGpuState->shutdown();
            mImpl->mEntitySceneGpuState.reset();
        }
        if (mImpl->mUltrasoundSystem)
        {
            mImpl->mUltrasoundSystem->shutdown();
            mImpl->mUltrasoundSystem.reset();
        }
        mImpl->mPhysicsSolver->shutdown();
        mImpl->mPhysicsSolver.reset();
        mImpl->mGpuDevice->shutdown();
        mImpl->mGpuDevice.reset();
        return false;
    }

    mImpl->mCustomComputeService = std::make_unique<CustomComputeService>(*mImpl->mGpuDevice);
    mImpl->mSharedBufferService  = std::make_unique<SharedBufferService>(*mImpl->mGpuDevice);
    mImpl->mInitialized          = true;
    return true;
}

void Runtime::shutdown()
{
    if (!mImpl->mInitialized)
    {
        return;
    }

    if (mImpl->mDeviceFrameActive && mImpl->mGpuDevice)
    {
        mImpl->mGpuDevice->endFrame(mImpl->mLastFrameContext);
        mImpl->mDeviceFrameActive = false;
    }

    mImpl->mRenderer.reset();
    if (mImpl->mCustomComputeService)
    {
        mImpl->mCustomComputeService->clear();
        mImpl->mCustomComputeService.reset();
    }
    if (mImpl->mSharedBufferService)
    {
        mImpl->mSharedBufferService->clear();
        mImpl->mSharedBufferService.reset();
    }

    if (mImpl->mRenderSceneUploader)
    {
        mImpl->mRenderSceneUploader->shutdown();
        mImpl->mRenderSceneUploader.reset();
    }
    if (mImpl->mEntitySceneGpuState)
    {
        mImpl->mEntitySceneGpuState->shutdown();
        mImpl->mEntitySceneGpuState.reset();
    }
    if (mImpl->mUltrasoundSystem)
    {
        mImpl->mUltrasoundSystem->shutdown();
        mImpl->mUltrasoundSystem.reset();
    }

    if (mImpl->mPhysicsSolver)
    {
        mImpl->mPhysicsSolver->shutdown();
        mImpl->mPhysicsSolver.reset();
    }

    if (mImpl->mGpuDevice)
    {
        mImpl->mGpuDevice->shutdown();
        mImpl->mGpuDevice.reset();
    }

    mImpl->mLastRenderStats                           = {};
    mImpl->mRenderFrameOptions                        = {};
    mImpl->mLastFrameContext                          = {};
    mImpl->mDeviceFrameActive                         = false;
    mImpl->mWorldUploaded                             = false;
    mImpl->mPhysicsPosesNeedSync                      = false;
    mImpl->mInitialized                               = false;
    mImpl->mLastUploadedEntityPoseRevision            = 0u;
    mImpl->mLastUploadedRenderableMetadataRevision    = 0u;
    mImpl->mLastUploadedRenderableQueueInfoRevision   = 0u;
    mImpl->mLastUploadedSoftBodyVertexBindingRevision = 0u;
    mImpl->mLastUploadedCameraInputRevision           = 0u;
    mImpl->mLastUploadedLightInputRevision            = 0u;
    mImpl->mLastUploadedLocalLightSelectionRevision   = 0u;
}

void Runtime::prepare()
{
    if (!mImpl->mInitialized)
    {
        return;
    }

    mImpl->mWorld.ensureRenderStateUpToDate(mImpl->mResources);
    if (mImpl->mUltrasoundSystem)
    {
        if (!mImpl->mUltrasoundSystem->prepare(mImpl->mWorld))
        {
            CRESSIM_LOG_WARNING("Runtime: ultrasound prepare failed.");
        }
    }
    mImpl->mWorldUploaded        = false;
    mImpl->mPhysicsPosesNeedSync = false;
}

bool Runtime::uploadWorld()
{
    if (!mImpl->mInitialized)
    {
        return false;
    }

    bool uploaded = true;
    if (mImpl->mPhysicsSolver &&
        !mImpl->mPhysicsSolver->syncWorldState(mImpl->mWorld.physicsWorld()))
    {
        CRESSIM_LOG_ERROR("Runtime: physics world upload failed.");
        uploaded = false;
    }

    if (uploaded && mImpl->mEntitySceneGpuState && mImpl->mRenderSceneUploader &&
        !syncGpuScene(
            mImpl->mWorld, mImpl->mEntitySceneGpuState.get(), mImpl->mRenderSceneUploader.get(),
            mImpl->mPhysicsSolver.get(), false, mImpl->mLastUploadedEntityPoseRevision,
            mImpl->mLastUploadedRenderableMetadataRevision,
            mImpl->mLastUploadedRenderableQueueInfoRevision,
            mImpl->mLastUploadedSoftBodyVertexBindingRevision,
            mImpl->mLastUploadedCameraInputRevision, mImpl->mLastUploadedLightInputRevision,
            mImpl->mLastUploadedLocalLightSelectionRevision))
    {
        uploaded = false;
    }

    mImpl->mWorldUploaded        = uploaded;
    mImpl->mPhysicsPosesNeedSync = false;
    return uploaded;
}

bool Runtime::stepPhysics(const common::FrameContext &frameContext)
{
    if (!mImpl->mInitialized)
    {
        return false;
    }
    if (!mImpl->mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: stepPhysics() requires uploadWorld() after prepare() and "
                          "before execution.");
        return false;
    }

    mImpl->mLastFrameContext = frameContext;
    ensureDeviceFrameActive(mImpl->mGpuDevice.get(), frameContext, mImpl->mDeviceFrameActive);

    bool physicsStepSucceeded = true;
    if (mImpl->mPhysicsSolver)
    {
        physicsStepSucceeded =
            mImpl->mPhysicsSolver->step(frameContext, mImpl->mWorld.physicsWorld());
        if (!physicsStepSucceeded)
        {
            CRESSIM_LOG_ERROR("Runtime: physics step failed at frame ", frameContext.frameIndex,
                              " (dt=", frameContext.deltaSeconds, ").");
        }
    }
    if (physicsStepSucceeded)
    {
        mImpl->mPhysicsPosesNeedSync = true;
    }
    return physicsStepSucceeded;
}

bool Runtime::stepSimulationSensors(const common::FrameContext &frameContext)
{
    if (!mImpl->mInitialized)
    {
        return false;
    }

    if (syncGpuScene(
            mImpl->mWorld, mImpl->mEntitySceneGpuState.get(), mImpl->mRenderSceneUploader.get(),
            mImpl->mPhysicsSolver.get(), mImpl->mPhysicsPosesNeedSync,
            mImpl->mLastUploadedEntityPoseRevision, mImpl->mLastUploadedRenderableMetadataRevision,
            mImpl->mLastUploadedRenderableQueueInfoRevision,
            mImpl->mLastUploadedSoftBodyVertexBindingRevision,
            mImpl->mLastUploadedCameraInputRevision, mImpl->mLastUploadedLightInputRevision,
            mImpl->mLastUploadedLocalLightSelectionRevision))
    {
        mImpl->mPhysicsPosesNeedSync = false;
    }

    mImpl->mLastFrameContext = frameContext;

    if (!mImpl->mUltrasoundSystem)
    {
        return true;
    }

    ensureDeviceFrameActive(mImpl->mGpuDevice.get(), frameContext, mImpl->mDeviceFrameActive);

    const bool succeeded = mImpl->mUltrasoundSystem->execute(frameContext, mImpl->mWorld);
    if (!succeeded)
    {
        CRESSIM_LOG_WARNING("Runtime: ultrasound step failed at frame ", frameContext.frameIndex,
                            ".");
    }
    return succeeded;
}

void Runtime::stepVisualSensors(const common::FrameContext &frameContext)
{
    if (!mImpl->mInitialized)
    {
        return;
    }

    const bool gpuSceneReady = syncGpuScene(
        mImpl->mWorld, mImpl->mEntitySceneGpuState.get(), mImpl->mRenderSceneUploader.get(),
        mImpl->mPhysicsSolver.get(), mImpl->mPhysicsPosesNeedSync,
        mImpl->mLastUploadedEntityPoseRevision, mImpl->mLastUploadedRenderableMetadataRevision,
        mImpl->mLastUploadedRenderableQueueInfoRevision,
        mImpl->mLastUploadedSoftBodyVertexBindingRevision, mImpl->mLastUploadedCameraInputRevision,
        mImpl->mLastUploadedLightInputRevision, mImpl->mLastUploadedLocalLightSelectionRevision);

    if (gpuSceneReady)
    {
        mImpl->mPhysicsPosesNeedSync = false;
    }

    // Wait once at the graphics consumer. Whichever sensor path runs first has already updated
    // the shared scene poses, so a later simulation-sensor call does not rewrite them.
    if (gpuSceneReady && mImpl->mGpuDevice && !mImpl->mGpuDevice->waitForPhysicsOnGraphics())
    {
        CRESSIM_LOG_WARNING("Runtime: failed to synchronize GPU scene for visual sensors.");
    }

    mImpl->mLastFrameContext = frameContext;

    ensureDeviceFrameActive(mImpl->mGpuDevice.get(), frameContext, mImpl->mDeviceFrameActive);

    physics::PhysicsGpuSceneView physicsSceneView{};
    const physics::PhysicsGpuSceneView *physicsScenePtr = nullptr;
    if (mImpl->mPhysicsSolver)
    {
        physicsSceneView = mImpl->mPhysicsSolver->gpuSceneView();
        physicsScenePtr  = &physicsSceneView;
    }

    mImpl->mLastRenderStats = mImpl->mRenderer->render(frameContext, mImpl->mWorld.hostSceneView(),
                                                       physicsScenePtr, mImpl->mRenderFrameOptions);
}

void Runtime::endFrame(const common::FrameContext &frameContext)
{
    if (!mImpl->mInitialized)
    {
        return;
    }

    mImpl->mLastFrameContext = frameContext;
    if (!mImpl->mDeviceFrameActive || !mImpl->mGpuDevice)
    {
        return;
    }

    mImpl->mGpuDevice->endFrame(frameContext);
    mImpl->mDeviceFrameActive = false;
}

World &Runtime::getWorld() noexcept
{
    return mImpl->mWorld;
}

const World &Runtime::getWorld() const noexcept
{
    return mImpl->mWorld;
}

gpu::GpuDevice *Runtime::getGpuDevice() noexcept
{
    return mImpl->mGpuDevice.get();
}

const gpu::GpuDevice *Runtime::getGpuDevice() const noexcept
{
    return mImpl->mGpuDevice.get();
}

physics::PhysicsSolver *Runtime::getPhysicsSolver() noexcept
{
    return mImpl->mPhysicsSolver.get();
}

const physics::PhysicsSolver *Runtime::getPhysicsSolver() const noexcept
{
    return mImpl->mPhysicsSolver.get();
}

void Runtime::setGravity(const Diligent::float3 &gravity) noexcept
{
    if (mImpl->mPhysicsSolver != nullptr)
    {
        mImpl->mPhysicsSolver->setGravity(gravity);
    }
}

const graphics::RenderStats &Runtime::lastRenderStats() const noexcept
{
    return mImpl->mLastRenderStats;
}

void Runtime::setRenderFrameOptions(const graphics::RenderFrameOptions &options) noexcept
{
    mImpl->mRenderFrameOptions = options;
}

const graphics::RenderFrameOptions &Runtime::renderFrameOptions() const noexcept
{
    return mImpl->mRenderFrameOptions;
}

graphics::RenderResourceManager &Runtime::getResources() noexcept
{
    return mImpl->mResources;
}

const graphics::RenderResourceManager &Runtime::getResources() const noexcept
{
    return mImpl->mResources;
}

RuntimeInfo Runtime::getInfo() const noexcept
{
    RuntimeInfo info{};
    info.engineVersion        = CRESSIM_NEO_VERSION;
    info.engineVersionMajor   = CRESSIM_NEO_VERSION_MAJOR;
    info.engineVersionMinor   = CRESSIM_NEO_VERSION_MINOR;
    info.engineVersionPatch   = CRESSIM_NEO_VERSION_PATCH;
    info.cudaInteropSupported = gpu::CudaSharedBuffer::supportsCudaInteropBuild();
#if CRESSIM_NEO_HAS_ULTRASOUND
    info.ultrasoundSupported = true;
#else
    info.ultrasoundSupported = false;
#endif
    return info;
}

SharedBufferHandle Runtime::createSharedBuffer(const SharedBufferDesc &desc)
{
    return mImpl->mInitialized && mImpl->mSharedBufferService != nullptr
               ? mImpl->mSharedBufferService->createBuffer(desc)
               : SharedBufferHandle{};
}

bool Runtime::destroySharedBuffer(const SharedBufferHandle handle)
{
    return mImpl->mInitialized && mImpl->mSharedBufferService != nullptr &&
           mImpl->mSharedBufferService->destroyBuffer(handle);
}

std::vector<SharedBufferInfo> Runtime::listSharedBuffers() const
{
    return mImpl->mInitialized && mImpl->mSharedBufferService != nullptr
               ? mImpl->mSharedBufferService->listBuffers()
               : std::vector<SharedBufferInfo>{};
}

bool Runtime::tryGetSharedBufferInfo(const SharedBufferHandle handle,
                                     SharedBufferInfo &outInfo) const
{
    return mImpl->mInitialized && mImpl->mSharedBufferService != nullptr &&
           mImpl->mSharedBufferService->tryGetBufferInfo(handle, outInfo);
}

bool Runtime::tryGetSharedBufferCudaView(const SharedBufferHandle handle,
                                         SharedBufferCudaView &outView) const
{
    return mImpl->mInitialized && mImpl->mSharedBufferService != nullptr &&
           mImpl->mSharedBufferService->tryGetCudaView(handle, outView);
}

SharedBufferLease Runtime::retainSharedBufferLease(const SharedBufferHandle handle) const
{
    return SharedBufferLease{mImpl->mInitialized && mImpl->mSharedBufferService != nullptr
                                 ? mImpl->mSharedBufferService->retainBuffer(handle)
                                 : std::shared_ptr<void>{}};
}

bool Runtime::syncSharedBufferToCuda(const SharedBufferHandle handle)
{
    if (!mImpl->mInitialized || mImpl->mSharedBufferService == nullptr ||
        mImpl->mGpuDevice == nullptr)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mImpl->mGpuDevice->tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        return false;
    }

    return mImpl->mSharedBufferService->synchronizeToCuda(handle, computeBackend.computeContext);
}

bool Runtime::syncSharedBufferFromCuda(const SharedBufferHandle handle)
{
    if (!mImpl->mInitialized || mImpl->mSharedBufferService == nullptr ||
        mImpl->mGpuDevice == nullptr)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mImpl->mGpuDevice->tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        return false;
    }

    return mImpl->mSharedBufferService->synchronizeFromCuda(handle, computeBackend.computeContext);
}

bool Runtime::tryGetPreparedRigidLayoutMapping(RigidLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mImpl->mInitialized || mImpl->mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld    = mImpl->mWorld.physicsWorld();
    const physics::RigidBodySoAHost &rigidBodies = physicsWorld.rigidBodySoA();
    const physics::ColliderSoAHost &colliders    = physicsWorld.colliderSoA();
    const physics::BodyColliderMappingHost &bodyColliderMapping =
        physicsWorld.bodyColliderMapping();
    outMapping.rigidBodyCount              = static_cast<std::uint32_t>(rigidBodies.size());
    outMapping.colliderCount               = static_cast<std::uint32_t>(colliders.size());
    outMapping.layoutRevision              = physicsWorld.rigidBodyTopologyRevision();
    outMapping.rigidBodyIds                = rigidBodies.rigidBodyIds;
    outMapping.rigidBodyEntityIds          = rigidBodies.entityIds;
    outMapping.rigidBodyEnvironmentIndices = rigidBodies.environmentIndices;
    outMapping.colliderIds                 = colliders.colliderIds;
    outMapping.colliderEntityIds           = colliders.entityIds;
    outMapping.colliderOwnerBodyIds        = colliders.ownerRigidBodyIds;
    outMapping.colliderOwnerBodyIndices    = colliders.ownerRigidBodyIndices;
    outMapping.colliderEnvironmentIndices  = colliders.environmentIndices;
    outMapping.colliderShapeTypes          = colliders.shapeTypes;
    outMapping.colliderEnabledFlags        = colliders.enabledFlags;
    outMapping.colliderCollisionLayers     = colliders.collisionLayers;
    outMapping.colliderCollisionMasks      = colliders.collisionMasks;
    outMapping.colliderShapeParams         = colliders.shapeParams;
    outMapping.bodyColliderOffsets         = bodyColliderMapping.colliderOffsets;
    outMapping.bodyColliderCounts          = bodyColliderMapping.colliderCounts;
    outMapping.bodyColliderIndices         = bodyColliderMapping.colliderIndices;
    outMapping.colliderLocalPositions.reserve(colliders.localPositions.size());
    outMapping.colliderLocalRotations.reserve(colliders.localOrientations.size());
    for (const Diligent::float4 &position : colliders.localPositions)
    {
        outMapping.colliderLocalPositions.push_back({position.x, position.y, position.z});
    }
    for (const Diligent::float4 &rotation : colliders.localOrientations)
    {
        outMapping.colliderLocalRotations.push_back(
            Diligent::QuaternionF{rotation.x, rotation.y, rotation.z, rotation.w});
    }
    return true;
}

bool Runtime::tryGetPreparedConstraintLayoutMapping(ConstraintLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mImpl->mInitialized || mImpl->mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld    = mImpl->mWorld.physicsWorld();
    const physics::RigidBodySoAHost &rigidBodies = physicsWorld.rigidBodySoA();
    const auto &rigidParticleAttachments = physicsWorld.rigidParticleAttachmentConstraintSnapshot();
    const auto &rigidDistanceConstraints = physicsWorld.rigidDistanceConstraintSnapshot();
    const auto &routedCables             = physicsWorld.routedCableConstraintSnapshot();

    std::unordered_map<common::EntityId, std::uint32_t> bodyIndexByEntity;
    bodyIndexByEntity.reserve(rigidBodies.entityIds.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(rigidBodies.entityIds.size()); ++i)
    {
        bodyIndexByEntity.emplace(rigidBodies.entityIds[i], i);
    }

    outMapping.layoutRevision = std::max({physicsWorld.rigidParticleAttachmentDefinitionRevision(),
                                          physicsWorld.rigidDistanceConstraintDefinitionRevision(),
                                          physicsWorld.routedCableDefinitionRevision()});

    auto &attachmentMapping = outMapping.rigidParticleAttachments;
    attachmentMapping.count = static_cast<std::uint32_t>(rigidParticleAttachments.size());
    attachmentMapping.constraintIds.reserve(rigidParticleAttachments.size());
    attachmentMapping.environmentIndices.reserve(rigidParticleAttachments.size());
    attachmentMapping.rigidBodyIds.reserve(rigidParticleAttachments.size());
    attachmentMapping.rigidBodyIndices.reserve(rigidParticleAttachments.size());
    attachmentMapping.particleEntityIds.reserve(rigidParticleAttachments.size());
    attachmentMapping.particleReferenceTypes.reserve(rigidParticleAttachments.size());
    attachmentMapping.particleLocalIndices.reserve(rigidParticleAttachments.size());
    attachmentMapping.enabledFlags.reserve(rigidParticleAttachments.size());
    for (const auto &constraint : rigidParticleAttachments)
    {
        const auto bodyIt = bodyIndexByEntity.find(constraint.rigidBodyEntityId);
        const std::uint32_t bodyIndex =
            bodyIt != bodyIndexByEntity.end() ? bodyIt->second : 0xffffffffu;
        attachmentMapping.constraintIds.push_back(constraint.constraintId);
        attachmentMapping.environmentIndices.push_back(
            bodyIndex < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndex]
                : 0u);
        attachmentMapping.rigidBodyIds.push_back(bodyIndex < rigidBodies.rigidBodyIds.size()
                                                     ? rigidBodies.rigidBodyIds[bodyIndex]
                                                     : physics::kInvalidRigidBodyId);
        attachmentMapping.rigidBodyIndices.push_back(bodyIndex);
        attachmentMapping.particleEntityIds.push_back(constraint.particle.entityId);
        attachmentMapping.particleReferenceTypes.push_back(
            static_cast<std::uint32_t>(constraint.particle.type));
        attachmentMapping.particleLocalIndices.push_back(constraint.particle.localParticleIndex);
        attachmentMapping.enabledFlags.push_back(constraint.enabled ? 1u : 0u);
    }

    auto &distanceMapping = outMapping.rigidDistanceConstraints;
    distanceMapping.count = static_cast<std::uint32_t>(rigidDistanceConstraints.size());
    distanceMapping.constraintIds.reserve(rigidDistanceConstraints.size());
    distanceMapping.environmentIndices.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIdsA.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIdsB.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIndicesA.reserve(rigidDistanceConstraints.size());
    distanceMapping.rigidBodyIndicesB.reserve(rigidDistanceConstraints.size());
    distanceMapping.enabledFlags.reserve(rigidDistanceConstraints.size());
    for (const auto &constraint : rigidDistanceConstraints)
    {
        const auto bodyItA = bodyIndexByEntity.find(constraint.entityA);
        const auto bodyItB = bodyIndexByEntity.find(constraint.entityB);
        const std::uint32_t bodyIndexA =
            bodyItA != bodyIndexByEntity.end() ? bodyItA->second : 0xffffffffu;
        const std::uint32_t bodyIndexB =
            bodyItB != bodyIndexByEntity.end() ? bodyItB->second : 0xffffffffu;
        distanceMapping.constraintIds.push_back(constraint.constraintId);
        distanceMapping.environmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        distanceMapping.rigidBodyIdsA.push_back(bodyIndexA < rigidBodies.rigidBodyIds.size()
                                                    ? rigidBodies.rigidBodyIds[bodyIndexA]
                                                    : physics::kInvalidRigidBodyId);
        distanceMapping.rigidBodyIdsB.push_back(bodyIndexB < rigidBodies.rigidBodyIds.size()
                                                    ? rigidBodies.rigidBodyIds[bodyIndexB]
                                                    : physics::kInvalidRigidBodyId);
        distanceMapping.rigidBodyIndicesA.push_back(bodyIndexA);
        distanceMapping.rigidBodyIndicesB.push_back(bodyIndexB);
        distanceMapping.enabledFlags.push_back(constraint.enabled ? 1u : 0u);
    }

    auto &routedCableMapping = outMapping.routedCables;
    routedCableMapping.count = static_cast<std::uint32_t>(routedCables.size());
    routedCableMapping.constraintIds.reserve(routedCables.size());
    routedCableMapping.environmentIndices.reserve(routedCables.size());
    routedCableMapping.routePointOffsets.reserve(routedCables.size());
    routedCableMapping.routePointCounts.reserve(routedCables.size());
    routedCableMapping.enabledFlags.reserve(routedCables.size());
    std::uint32_t routePointOffset = 0u;
    for (const auto &constraint : routedCables)
    {
        routedCableMapping.constraintIds.push_back(constraint.constraintId);
        routedCableMapping.routePointOffsets.push_back(routePointOffset);
        routedCableMapping.routePointCounts.push_back(
            static_cast<std::uint32_t>(constraint.routePoints.size()));
        routedCableMapping.enabledFlags.push_back(constraint.enabled ? 1u : 0u);

        std::uint32_t envIndex = 0u;
        bool haveEnv           = false;
        for (const auto &routePoint : constraint.routePoints)
        {
            const auto bodyIt = bodyIndexByEntity.find(routePoint.entityId);
            const std::uint32_t bodyIndex =
                bodyIt != bodyIndexByEntity.end() ? bodyIt->second : 0xffffffffu;
            if (!haveEnv && bodyIndex < rigidBodies.environmentIndices.size())
            {
                envIndex = rigidBodies.environmentIndices[bodyIndex];
                haveEnv  = true;
            }
            routedCableMapping.routePointRigidBodyIds.push_back(
                bodyIndex < rigidBodies.rigidBodyIds.size() ? rigidBodies.rigidBodyIds[bodyIndex]
                                                            : physics::kInvalidRigidBodyId);
            routedCableMapping.routePointRigidBodyIndices.push_back(bodyIndex);
            routedCableMapping.routePointLocalGuideOffsets.push_back(routePoint.localGuideOffset);
        }
        routedCableMapping.environmentIndices.push_back(envIndex);
        routePointOffset += static_cast<std::uint32_t>(constraint.routePoints.size());
    }

    return true;
}

bool Runtime::tryGetPreparedParticleLayoutMapping(ParticleLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mImpl->mInitialized || mImpl->mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld = mImpl->mWorld.physicsWorld();
    physicsWorld.ensureDerivedStateUpToDate();

    const physics::ParticleSoAHost &particles = physicsWorld.particles();
    const auto &softBodies                    = physicsWorld.softBodySnapshot();
    const auto &fluids                        = physicsWorld.fluidSnapshot();
    const auto &strands                       = physicsWorld.strandSnapshot();
    outMapping.particleCount                  = static_cast<std::uint32_t>(particles.size());
    outMapping.softBodyCount                  = static_cast<std::uint32_t>(softBodies.size());
    outMapping.fluidCount                     = static_cast<std::uint32_t>(fluids.size());
    outMapping.strandCount                    = static_cast<std::uint32_t>(strands.size());
    outMapping.layoutRevision                 = physicsWorld.softParticleRevision();
    outMapping.environmentIndices             = particles.environmentIndices;
    outMapping.particleKinds                  = particles.particleKinds;
    outMapping.ownerTypes                     = particles.ownerTypes;
    outMapping.ownerIndices                   = particles.ownerIndices;
    outMapping.strandIds                      = particles.strandIds;
    outMapping.strandOrders                   = particles.strandOrders;
    outMapping.strandRoles                    = particles.strandRoles;
    outMapping.owningSoftBodyIndices          = particles.owningSoftBodyIndices;
    outMapping.particleMaterialIndices        = particles.particleMaterialIndices;
    outMapping.fluidMaterialIndices           = particles.fluidMaterialIndices;
    outMapping.phases                         = particles.phases;
    outMapping.collisionLayers                = particles.collisionLayers;
    outMapping.collisionMasks                 = particles.collisionMasks;
    outMapping.adjacencyOffsets               = particles.adjacencyOffsets;
    outMapping.adjacencyCounts                = particles.adjacencyCounts;

    outMapping.softBodyEntityIds.reserve(softBodies.size());
    outMapping.softBodyEnvironmentIndices.reserve(softBodies.size());
    outMapping.softBodyParticleOffsets.reserve(softBodies.size());
    outMapping.softBodyParticleCounts.reserve(softBodies.size());
    for (const auto &softBody : softBodies)
    {
        outMapping.softBodyEntityIds.push_back(softBody.entityId);
        outMapping.softBodyEnvironmentIndices.push_back(softBody.environmentIndex);
        outMapping.softBodyParticleOffsets.push_back(softBody.particleOffset);
        outMapping.softBodyParticleCounts.push_back(softBody.particleCount);
    }

    outMapping.fluidEntityIds.reserve(fluids.size());
    outMapping.fluidEnvironmentIndices.reserve(fluids.size());
    outMapping.fluidParticleOffsets.reserve(fluids.size());
    outMapping.fluidParticleCounts.reserve(fluids.size());
    for (const auto &fluid : fluids)
    {
        outMapping.fluidEntityIds.push_back(fluid.entityId);
        outMapping.fluidEnvironmentIndices.push_back(fluid.environmentIndex);
        outMapping.fluidParticleOffsets.push_back(fluid.particleOffset);
        outMapping.fluidParticleCounts.push_back(fluid.particleCount);
    }

    outMapping.strandEntityIds.reserve(strands.size());
    outMapping.strandEnvironmentIndices.reserve(strands.size());
    outMapping.strandParticleOffsets.reserve(strands.size());
    outMapping.strandParticleCounts.reserve(strands.size());
    for (const auto &strand : strands)
    {
        outMapping.strandEntityIds.push_back(strand.entityId);
        outMapping.strandEnvironmentIndices.push_back(strand.environmentIndex);
        outMapping.strandParticleOffsets.push_back(strand.particleOffset);
        outMapping.strandParticleCounts.push_back(strand.particleCount);
    }

    return true;
}

bool Runtime::tryGetPreparedJointLayoutMapping(JointLayoutMapping &outMapping) const
{
    outMapping = {};
    if (!mImpl->mInitialized || mImpl->mPhysicsSolver == nullptr)
    {
        return false;
    }

    const physics::PhysicsWorld &physicsWorld           = mImpl->mWorld.physicsWorld();
    const physics::RigidBodySoAHost &rigidBodies        = physicsWorld.rigidBodySoA();
    const std::vector<physics::BallJointState> &balls   = physicsWorld.ballJointSnapshot();
    const std::vector<physics::HingeJointState> &hinges = physicsWorld.hingeJointSnapshot();
    const std::vector<physics::SphericalJointState> &spherical =
        physicsWorld.sphericalJointSnapshot();
    const std::vector<physics::SliderJointState> &sliders = physicsWorld.sliderJointSnapshot();
    const physics::RigidJointSceneHost &jointScene        = physicsWorld.rigidJointScene();

    outMapping.ballJointCount      = static_cast<std::uint32_t>(balls.size());
    outMapping.hingeJointCount     = static_cast<std::uint32_t>(hinges.size());
    outMapping.sphericalJointCount = static_cast<std::uint32_t>(spherical.size());
    outMapping.sliderJointCount    = static_cast<std::uint32_t>(sliders.size());
    outMapping.layoutRevision      = physicsWorld.rigidJointTopologyRevision();

    outMapping.ballJointIds.reserve(balls.size());
    outMapping.ballEnvironmentIndices.reserve(balls.size());
    outMapping.ballBodyIdsA.reserve(balls.size());
    outMapping.ballBodyIdsB.reserve(balls.size());
    outMapping.ballBodyIndicesA.reserve(balls.size());
    outMapping.ballBodyIndicesB.reserve(balls.size());
    for (std::size_t i = 0; i < balls.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.ball.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.ball.bodyIndicesB[i];
        outMapping.ballJointIds.push_back(balls[i].jointId);
        outMapping.ballEnvironmentIndices.push_back(bodyIndexA <
                                                            rigidBodies.environmentIndices.size()
                                                        ? rigidBodies.environmentIndices[bodyIndexA]
                                                        : 0u);
        outMapping.ballBodyIdsA.push_back(balls[i].bodyA);
        outMapping.ballBodyIdsB.push_back(balls[i].bodyB);
        outMapping.ballBodyIndicesA.push_back(bodyIndexA);
        outMapping.ballBodyIndicesB.push_back(bodyIndexB);
    }

    outMapping.hingeJointIds.reserve(hinges.size());
    outMapping.hingeEnvironmentIndices.reserve(hinges.size());
    outMapping.hingeBodyIdsA.reserve(hinges.size());
    outMapping.hingeBodyIdsB.reserve(hinges.size());
    outMapping.hingeBodyIndicesA.reserve(hinges.size());
    outMapping.hingeBodyIndicesB.reserve(hinges.size());
    for (std::size_t i = 0; i < hinges.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.hinge.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.hinge.bodyIndicesB[i];
        outMapping.hingeJointIds.push_back(hinges[i].jointId);
        outMapping.hingeEnvironmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        outMapping.hingeBodyIdsA.push_back(hinges[i].bodyA);
        outMapping.hingeBodyIdsB.push_back(hinges[i].bodyB);
        outMapping.hingeBodyIndicesA.push_back(bodyIndexA);
        outMapping.hingeBodyIndicesB.push_back(bodyIndexB);
    }

    outMapping.sphericalJointIds.reserve(spherical.size());
    outMapping.sphericalEnvironmentIndices.reserve(spherical.size());
    outMapping.sphericalBodyIdsA.reserve(spherical.size());
    outMapping.sphericalBodyIdsB.reserve(spherical.size());
    outMapping.sphericalBodyIndicesA.reserve(spherical.size());
    outMapping.sphericalBodyIndicesB.reserve(spherical.size());
    for (std::size_t i = 0; i < spherical.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.spherical.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.spherical.bodyIndicesB[i];
        outMapping.sphericalJointIds.push_back(spherical[i].jointId);
        outMapping.sphericalEnvironmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        outMapping.sphericalBodyIdsA.push_back(spherical[i].bodyA);
        outMapping.sphericalBodyIdsB.push_back(spherical[i].bodyB);
        outMapping.sphericalBodyIndicesA.push_back(bodyIndexA);
        outMapping.sphericalBodyIndicesB.push_back(bodyIndexB);
    }

    outMapping.sliderJointIds.reserve(sliders.size());
    outMapping.sliderEnvironmentIndices.reserve(sliders.size());
    outMapping.sliderBodyIdsA.reserve(sliders.size());
    outMapping.sliderBodyIdsB.reserve(sliders.size());
    outMapping.sliderBodyIndicesA.reserve(sliders.size());
    outMapping.sliderBodyIndicesB.reserve(sliders.size());
    for (std::size_t i = 0; i < sliders.size(); ++i)
    {
        const std::uint32_t bodyIndexA = jointScene.slider.bodyIndicesA[i];
        const std::uint32_t bodyIndexB = jointScene.slider.bodyIndicesB[i];
        outMapping.sliderJointIds.push_back(sliders[i].jointId);
        outMapping.sliderEnvironmentIndices.push_back(
            bodyIndexA < rigidBodies.environmentIndices.size()
                ? rigidBodies.environmentIndices[bodyIndexA]
                : 0u);
        outMapping.sliderBodyIdsA.push_back(sliders[i].bodyA);
        outMapping.sliderBodyIdsB.push_back(sliders[i].bodyB);
        outMapping.sliderBodyIndicesA.push_back(bodyIndexA);
        outMapping.sliderBodyIndicesB.push_back(bodyIndexB);
    }

    return true;
}

bool Runtime::computeUltrasoundProbeLayout(const UltrasoundProbeComponent &probeComponent,
                                           const UltrasoundRendererComponent &rendererComponent,
                                           UltrasoundProbeLayout &outLayout) const
{
    outLayout = {};
    return mImpl->mInitialized && mImpl->mUltrasoundSystem != nullptr &&
           mImpl->mUltrasoundSystem->computeProbeLayout(probeComponent, rendererComponent,
                                                        outLayout);
}

std::vector<CustomComputeResourceDesc> Runtime::listCustomComputeResources()
{
    if (!mImpl->mInitialized || mImpl->mCustomComputeService == nullptr ||
        mImpl->mPhysicsSolver == nullptr)
    {
        return {};
    }
    if (!mImpl->mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: listCustomComputeResources() requires uploadWorld() after "
                          "prepare() and before execution.");
        return {};
    }
    return mImpl->mCustomComputeService->listResources(
        *mImpl->mPhysicsSolver, mImpl->mWorld.physicsWorld(), mImpl->mWorld.gpuEntityScene());
}

CustomComputePassHandle Runtime::createCustomComputePass(const CustomComputePassDesc &desc)
{
    if (!mImpl->mInitialized || mImpl->mCustomComputeService == nullptr ||
        mImpl->mPhysicsSolver == nullptr)
    {
        return {};
    }
    if (!mImpl->mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: createCustomComputePass() requires uploadWorld() after "
                          "prepare() and before execution.");
        return {};
    }
    ensureDeviceFrameActive(mImpl->mGpuDevice.get(), mImpl->mLastFrameContext,
                            mImpl->mDeviceFrameActive);
    return mImpl->mCustomComputeService->createPass(
        *mImpl->mPhysicsSolver, mImpl->mWorld.physicsWorld(), mImpl->mWorld.gpuEntityScene(),
        mImpl->mSharedBufferService.get(), desc);
}

bool Runtime::updateCustomComputePassConstants(CustomComputePassHandle handle,
                                               const std::vector<std::uint8_t> &data)
{
    if (!mImpl->mInitialized || mImpl->mCustomComputeService == nullptr)
    {
        return false;
    }
    ensureDeviceFrameActive(mImpl->mGpuDevice.get(), mImpl->mLastFrameContext,
                            mImpl->mDeviceFrameActive);
    return mImpl->mCustomComputeService->updatePassConstants(handle, data);
}

bool Runtime::executeCustomComputePass(CustomComputePassHandle handle)
{
    if (!mImpl->mInitialized || mImpl->mCustomComputeService == nullptr ||
        mImpl->mPhysicsSolver == nullptr)
    {
        return false;
    }
    if (!mImpl->mWorldUploaded)
    {
        CRESSIM_LOG_ERROR("Runtime: executeCustomComputePass() requires uploadWorld() after "
                          "prepare() and before execution.");
        return false;
    }
    ensureDeviceFrameActive(mImpl->mGpuDevice.get(), mImpl->mLastFrameContext,
                            mImpl->mDeviceFrameActive);
    return mImpl->mCustomComputeService->executePass(
        *mImpl->mPhysicsSolver, mImpl->mWorld.physicsWorld(), mImpl->mWorld.gpuEntityScene(),
        mImpl->mSharedBufferService.get(), handle);
}

bool Runtime::destroyCustomComputePass(CustomComputePassHandle handle)
{
    return mImpl->mInitialized && mImpl->mCustomComputeService != nullptr &&
           mImpl->mCustomComputeService->destroyPass(handle);
}

} // namespace cressim::neo::engine
