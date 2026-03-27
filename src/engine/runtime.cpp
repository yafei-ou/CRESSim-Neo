#include "engine/runtime.h"

#include "common/logger.h"

#include <sstream>

namespace cressim::neo::engine
{

namespace
{

const char *stageName(physics::PhysicsSolverStage stage)
{
    switch (stage)
    {
    case physics::PhysicsSolverStage::PredictState:
        return "PredictState";
    case physics::PhysicsSolverStage::UpdateWorldAabbs:
        return "UpdateWorldAabbs";
    case physics::PhysicsSolverStage::BuildBroadPhase:
        return "BuildBroadPhase";
    case physics::PhysicsSolverStage::GenerateBroadPhasePairs:
        return "GenerateBroadPhasePairs";
    case physics::PhysicsSolverStage::GenerateContacts:
        return "GenerateContacts";
    case physics::PhysicsSolverStage::SolveConstraints:
        return "SolveConstraints";
    case physics::PhysicsSolverStage::UpdateVelocities:
        return "UpdateVelocities";
    case physics::PhysicsSolverStage::CommitResults:
        return "CommitResults";
    case physics::PhysicsSolverStage::Count:
        break;
    }
    return "Unknown";
}

void logPhysicsStepFailure(const common::FrameContext &frameContext,
                           const physics::PhysicsSolverStageStats &stats)
{
    std::ostringstream stream;
    stream << "Runtime: physics step failed at frame " << frameContext.frameIndex
           << " (dt=" << frameContext.deltaSeconds << "). Executed stages:";
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(physics::PhysicsSolverStage::Count);
         ++i)
    {
        const auto stage = static_cast<physics::PhysicsSolverStage>(i);
        stream << ' ' << stageName(stage) << '=' << (stats.executed[i] ? '1' : '0');
    }
    CRESSIM_LOG_ERROR(stream.str());
}

} // namespace

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

    mGpuSceneSync = std::make_unique<gpu::GpuSceneSync>(*mGpuDevice);
    if (!mGpuSceneSync || !mGpuSceneSync->initialize(config.sceneLayout))
    {
        mGpuSceneSync.reset();
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
        mGpuSceneSync->shutdown();
        mGpuSceneSync.reset();
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

    if (mGpuSceneSync)
    {
        mGpuSceneSync->shutdown();
        mGpuSceneSync.reset();
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
            logPhysicsStepFailure(frameContext, mPhysicsSolver->lastStageStats());
        }
    }

    mWorld.ensureRenderStateUpToDate(mResources);

    bool gpuSceneReady = false;
    if (mGpuSceneSync)
    {
        gpuSceneReady = mGpuSceneSync->syncEntityPoseData(mWorld.renderObjectPositions(),
                                                          mWorld.renderObjectOrientations(),
                                                          mWorld.renderObjectScales());
    }
    if (gpuSceneReady && physicsStepSucceeded && mGpuSceneSync && mPhysicsSolver)
    {
        if (mGpuSceneSync->syncEntityPoses(mPhysicsSolver->gpuSceneView().rigid.poses,
                                           mWorld.physicsRenderableMappings()))
        {
        }
        else
        {
            gpuSceneReady = false;
        }
    }
    if (gpuSceneReady && mGpuDevice && !mGpuDevice->synchronizePhysicsToGraphics())
    {
        gpuSceneReady = false;
    }
    if (gpuSceneReady && mGpuSceneSync)
    {
        if (mGpuSceneSync->syncRenderableMetadata(mWorld.renderableMetadata()) &&
            mGpuSceneSync->syncRenderableQueueInfo(mWorld.renderableQueueInfo()) &&
            mGpuSceneSync->syncCameraInputs(mWorld.cameraInputs()) &&
            mGpuSceneSync->syncLightInputs(mWorld.lightInputs()) &&
            mGpuSceneSync->syncLocalLightSelections(mWorld.localLightSelections()))
        {
            mWorld.setGpuEntityScene(mGpuSceneSync->sceneView());
        }
        else
        {
            mWorld.setGpuEntityScene({});
        }
    }
    else
    {
        mWorld.setGpuEntityScene({});
    }

    mLastRenderStats = mRenderer->render(frameContext, mWorld.hostSceneView(), mRenderFrameOptions);
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
