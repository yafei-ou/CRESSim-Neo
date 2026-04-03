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

    mWorld.setSceneLayout(config.sceneLayout);

    mRenderSceneUploader = std::make_unique<RenderSceneUploader>(*mGpuDevice);
    if (!mRenderSceneUploader || !mRenderSceneUploader->initialize(config.sceneLayout))
    {
        mRenderSceneUploader.reset();
        mPhysicsSolver->shutdown();
        mPhysicsSolver.reset();
        mGpuDevice->shutdown();
        mGpuDevice.reset();
        return false;
    }

    mRenderer = std::make_unique<graphics::Renderer>(*mGpuDevice, mResources, config.rendererDesc);
    if (!mRenderer->initialize())
    {
        mRenderer.reset();
        mRenderSceneUploader->shutdown();
        mRenderSceneUploader.reset();
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
    mInitialized        = false;
}

void Runtime::tick(const common::FrameContext &frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    bool physicsStepSucceeded = true;
    if (mPhysicsSolver)
    {
        physicsStepSucceeded = mPhysicsSolver->step(frameContext, mWorld.physicsWorld());
        if (!physicsStepSucceeded)
        {
            logPhysicsStepFailure(frameContext);
        }
    }

    mWorld.ensureRenderStateUpToDate(mResources);

    bool gpuSceneReady = false;
    if (mRenderSceneUploader)
    {
        gpuSceneReady = mRenderSceneUploader->uploadEntityPoseData(
            mWorld.renderObjectPositions(), mWorld.renderObjectOrientations(),
            mWorld.renderObjectScales());
    }
    if (gpuSceneReady && physicsStepSucceeded && mRenderSceneUploader && mPhysicsSolver)
    {
        if (!mRenderSceneUploader->applyMappedEntityPoses(
                mPhysicsSolver->gpuSceneView().rigid.poses, mWorld.physicsRenderableMappings()))
        {
            gpuSceneReady = false;
        }
    }
    if (gpuSceneReady && mRenderSceneUploader &&
        mRenderSceneUploader->uploadRenderableMetadata(mWorld.renderableMetadata()) &&
        mRenderSceneUploader->uploadRenderableQueueInfo(mWorld.renderableQueueInfo()) &&
        mRenderSceneUploader->uploadCameraInputs(mWorld.cameraInputs()) &&
        mRenderSceneUploader->uploadLightInputs(mWorld.lightInputs()) &&
        mRenderSceneUploader->uploadLocalLightSelections(mWorld.localLightSelections()) &&
        (!mGpuDevice || mGpuDevice->waitForPhysicsOnGraphics()))
    {
        mWorld.setGpuEntityScene(mRenderSceneUploader->sceneView());
    }
    else
    {
        CRESSIM_LOG_WARNING("Runtime: GPU scene sync failed.");
        mWorld.setGpuEntityScene({});
    }

    physics::PhysicsGpuSceneView physicsSceneView{};
    const physics::PhysicsGpuSceneView *physicsScenePtr = nullptr;
    if (mPhysicsSolver)
    {
        physicsSceneView = mPhysicsSolver->gpuSceneView();
        physicsScenePtr  = &physicsSceneView;
    }

    mLastRenderStats =
        mRenderer->render(frameContext, mWorld.hostSceneView(), physicsScenePtr,
                          mRenderFrameOptions);
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
