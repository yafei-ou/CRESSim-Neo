#include "engine/runtime.h"

#include "common/logger.h"

namespace cressim::neo::engine
{

namespace
{

void logPhysicsStepFailure(const common::FrameContext &frameContext)
{
    CRESSIM_LOG_ERROR("Runtime: physics step failed at frame ", frameContext.frameIndex,
                      " (dt=", frameContext.deltaSeconds, ").");
}

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

    mInitialized = true;
    return true;
}

void Runtime::shutdown()
{
    if (!mInitialized)
    {
        return;
    }

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

    mLastRenderStats      = {};
    mRenderFrameOptions   = {};
    mLastFrameContext     = {};
    mFrameBoundaryPending = false;
    mHasPhysicsState      = false;
    mInitialized          = false;
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
    mHasPhysicsState = false;
}

bool Runtime::stepPhysics(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return false;
    }

    mLastFrameContext = frameContext;

    bool physicsStepSucceeded = true;
    if (mPhysicsSolver)
    {
        physicsStepSucceeded = mPhysicsSolver->step(frameContext, mWorld.physicsWorld());
        if (!physicsStepSucceeded)
        {
            logPhysicsStepFailure(frameContext);
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

    const bool succeeded = mUltrasoundSystem->execute(frameContext, mWorld);
    if (!succeeded)
    {
        CRESSIM_LOG_WARNING("Runtime: ultrasound step failed at frame ", frameContext.frameIndex,
                            ".");
    }
    else
    {
        mFrameBoundaryPending = true;
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

    physics::PhysicsGpuSceneView physicsSceneView{};
    const physics::PhysicsGpuSceneView *physicsScenePtr = nullptr;
    if (mPhysicsSolver)
    {
        physicsSceneView = mPhysicsSolver->gpuSceneView();
        physicsScenePtr  = &physicsSceneView;
    }

    mLastRenderStats      = mRenderer->render(frameContext, mWorld.hostSceneView(), physicsScenePtr,
                                              mRenderFrameOptions);
    mFrameBoundaryPending = false;
}

void Runtime::flushSimulationSensors()
{
    if (!mInitialized || !mFrameBoundaryPending || !mGpuDevice)
    {
        return;
    }

    mGpuDevice->beginFrame(mLastFrameContext);
    mGpuDevice->endFrame(mLastFrameContext);
    mFrameBoundaryPending = false;
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

} // namespace cressim::neo::engine
