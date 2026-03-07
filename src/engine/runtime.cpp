#include "engine/runtime.h"

#include "engine/physics_world_to_world_sync.h"
#include "engine/world_to_physics_world_sync.h"
#include "engine/world_to_render_world_sync.h"

#include <iostream>

namespace cressim::neo::engine
{

namespace
{

const char* stageName(physics::PhysicsSolverStage stage)
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

void logPhysicsStepFailure(const common::FrameContext& frameContext,
                           const physics::PhysicsSolverStageStats& stats)
{
    std::cerr << "Runtime: physics step failed at frame " << frameContext.frameIndex
              << " (dt=" << frameContext.deltaSeconds << "). Executed stages:";
    for (std::uint32_t i = 0;
         i < static_cast<std::uint32_t>(physics::PhysicsSolverStage::Count); ++i)
    {
        const auto stage = static_cast<physics::PhysicsSolverStage>(i);
        std::cerr << ' ' << stageName(stage) << '=' << (stats.executed[i] ? '1' : '0');
    }
    std::cerr << '\n';
}

} // namespace

bool Runtime::initialize(const RuntimeConfig& config)
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

    mRenderer = std::make_unique<graphics::Renderer>(*mGpuDevice, mResources, config.rendererDesc);
    if (!mRenderer->initialize())
    {
        mRenderer.reset();
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

    mLastRenderStats                = {};
    mLastSyncedWorldRevision        = ~0ull;
    mLastSyncedPhysicsWorldRevision = ~0ull;
    mInitialized                    = false;
}

void Runtime::tick(const common::FrameContext& frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    (void)syncWorldToPhysicsWorld();
    bool physicsStepSucceeded = true;
    if (mPhysicsSolver)
    {
        physicsStepSucceeded = mPhysicsSolver->step(frameContext, mPhysicsWorld);
        if (!physicsStepSucceeded)
        {
            logPhysicsStepFailure(frameContext, mPhysicsSolver->lastStageStats());
        }
    }
    if (physicsStepSucceeded)
    {
        (void)syncPhysicsWorldToWorld();
    }
    const bool syncSkipped = syncWorldToRenderWorld();

    mLastRenderStats                        = mRenderer->render(frameContext, mRenderWorld);
    mLastRenderStats.worldSyncSkippedFrames = syncSkipped ? 1u : 0u;
}

World& Runtime::getWorld() noexcept
{
    return mWorld;
}

const World& Runtime::getWorld() const noexcept
{
    return mWorld;
}

gpu::GpuDevice* Runtime::getGpuDevice() noexcept
{
    return mGpuDevice.get();
}

const gpu::GpuDevice* Runtime::getGpuDevice() const noexcept
{
    return mGpuDevice.get();
}

physics::PhysicsSolver* Runtime::getPhysicsSolver() noexcept
{
    return mPhysicsSolver.get();
}

const physics::PhysicsSolver* Runtime::getPhysicsSolver() const noexcept
{
    return mPhysicsSolver.get();
}

const graphics::RenderStats& Runtime::lastRenderStats() const noexcept
{
    return mLastRenderStats;
}

graphics::RenderResourceManager& Runtime::getResources() noexcept
{
    return mResources;
}

const graphics::RenderResourceManager& Runtime::getResources() const noexcept
{
    return mResources;
}

bool Runtime::syncWorldToPhysicsWorld()
{
    if (mWorld.revision() == mLastSyncedWorldRevision)
    {
        return true;
    }

    detail::syncWorldToPhysicsWorld(mWorld, mPhysicsWorld);
    return false;
}

bool Runtime::syncPhysicsWorldToWorld()
{
    detail::syncPhysicsWorldToWorld(mPhysicsWorld, mWorld);
    mLastSyncedPhysicsWorldRevision = mPhysicsWorld.revision();
    return true;
}

bool Runtime::syncWorldToRenderWorld()
{
    if (mWorld.revision() == mLastSyncedWorldRevision)
    {
        return true;
    }

    detail::syncWorldToRenderWorld(mWorld, mRenderWorld);
    mWorld.clearDirtyEntities();
    mLastSyncedWorldRevision = mWorld.revision();
    return false;
}

} // namespace cressim::neo::engine
