#include "physics/physics_solver.h"

#include "physics/physics_pass_dispatcher.h"
#include "physics/physics_scene_gpu_state.h"

#include "DiligentEngine/DiligentCore/Primitives/interface/Errors.hpp"

#include <algorithm>
#include <memory>

namespace cressim::neo::physics
{

namespace
{

constexpr std::size_t stageIndex(PhysicsSolverStage stage)
{
    return static_cast<std::size_t>(stage);
}

void markStage(PhysicsSolverStageStats& stats, PhysicsSolverStage stage, bool executed)
{
    stats.executed[stageIndex(stage)] = executed;
    if (executed)
    {
        ++stats.dispatchedStages;
    }
    else
    {
        ++stats.skippedStages;
    }
}

void markAllStagesSkipped(PhysicsSolverStageStats& stats)
{
    markStage(stats, PhysicsSolverStage::PredictState, false);
    markStage(stats, PhysicsSolverStage::UpdateWorldAabbs, false);
    markStage(stats, PhysicsSolverStage::BuildBroadPhase, false);
    markStage(stats, PhysicsSolverStage::GenerateBroadPhasePairs, false);
    markStage(stats, PhysicsSolverStage::GenerateContacts, false);
    markStage(stats, PhysicsSolverStage::SolveConstraints, false);
    markStage(stats, PhysicsSolverStage::UpdateVelocities, false);
    markStage(stats, PhysicsSolverStage::CommitResults, false);
}

} // namespace

struct PhysicsSolver::Impl
{
    PhysicsSceneGpuState sceneState;
    PhysicsPassDispatcher passDispatcher;
    PhysicsSolverStageStats stageStats{};
};

PhysicsSolver::PhysicsSolver(gpu::GpuDevice& device, const PhysicsSolverDesc& desc)
    : mDevice(device), mDesc(desc), mImpl(std::make_unique<Impl>())
{
}

PhysicsSolver::~PhysicsSolver() = default;

bool PhysicsSolver::initialize()
{
    shutdown();

    if (!mDesc.enableGpuCompute)
    {
        mInitialized = true;
        return true;
    }

    gpu::GpuComputeBackendContext computeContext{};
    if (!mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to get physics GPU context.");
        return false;
    }

    mImpl = std::make_unique<Impl>();
    if (!mImpl->passDispatcher.initialize(computeContext.renderDevice, computeContext.contextId,
                                          mDevice.shaderSourceDirectory().c_str()))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver: failed to initialize physics pass dispatcher.");
        return false;
    }

    mInitialized = true;
    return true;
}

void PhysicsSolver::shutdown()
{
    mImpl        = std::make_unique<Impl>();
    mInitialized = false;
}

bool PhysicsSolver::step(const common::FrameContext& frameContext, PhysicsWorld& world)
{
    if (!mInitialized)
    {
        return false;
    }

    mImpl->stageStats = PhysicsSolverStageStats{};

    if (!mDesc.enableGpuCompute)
    {
        markAllStagesSkipped(mImpl->stageStats);
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: missing physics backend context.");
        return false;
    }

    const std::uint32_t rigidBodyCount = world.rigidBodyCount();
    const std::uint32_t colliderCount  = world.colliderCount();
    if (rigidBodyCount == 0u)
    {
        markAllStagesSkipped(mImpl->stageStats);
        return true;
    }

    if (!mImpl->sceneState.ensureCapacity(computeBackend.renderDevice, rigidBodyCount,
                                          colliderCount, computeBackend.contextId))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: ensureCapacity.");
        return false;
    }
    if (!mImpl->sceneState.uploadWorldState(computeBackend.computeContext, world, rigidBodyCount,
                                            colliderCount))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: uploadWorldState.");
        return false;
    }

    const std::uint32_t substeps   = std::max<std::uint32_t>(mDesc.substeps, 1u);
    const std::uint32_t iterations = std::max<std::uint32_t>(mDesc.solverIterations, 1u);
    const float substepDt          = frameContext.deltaSeconds / static_cast<float>(substeps);

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        GpuRigidDispatchConstants constants{};
        constants.dt                    = substepDt;
        constants.rigidBodyCount        = rigidBodyCount;
        constants.colliderCount         = colliderCount;
        constants.candidatePairCapacity = mImpl->sceneState.candidatePairCapacity();
        constants.substepIndex          = substep;
        constants.solverIterations      = iterations;

        if (mImpl->sceneState.correctionBuffersNeedClear() &&
            !mImpl->passDispatcher.clearCorrections(computeBackend.computeContext,
                                                    mImpl->sceneState, rigidBodyCount, constants))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: ClearCorrections dispatch.");
            return false;
        }

        if (!mImpl->passDispatcher.predict(computeBackend.computeContext, mImpl->sceneState,
                                           rigidBodyCount, constants))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: PredictState dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, true);

        if (!mImpl->passDispatcher.updateWorldAabbs(computeBackend.computeContext,
                                                    mImpl->sceneState, colliderCount, constants))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: UpdateWorldAabbs dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateWorldAabbs, true);

        if (colliderCount > 0u &&
            !mImpl->passDispatcher.compactBroadPhaseBodySets(
                computeBackend.computeContext, mImpl->sceneState, colliderCount, constants))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: BuildBroadPhase compaction dispatch.");
            return false;
        }

        GpuBroadPhaseMeta broadPhaseMeta{};
        if (colliderCount > 0u && !mImpl->sceneState.readbackBroadPhaseMetaBlocking(
                                      computeBackend.computeContext, broadPhaseMeta))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: readbackBroadPhaseMetaBlocking.");
            return false;
        }

        const std::uint32_t activeMovingCount =
            colliderCount > 0u ? broadPhaseMeta.activeMovingCount : 0u;
        constants.activeMovingCount = activeMovingCount;
        constants.staticBodyCount   = colliderCount > 0u ? broadPhaseMeta.staticBodyCount : 0u;
        bool builtBroadPhase        = false;
        if (activeMovingCount > 0u)
        {
            if (!mImpl->passDispatcher.buildBroadPhase(
                    computeBackend.computeContext, mImpl->sceneState, activeMovingCount, constants))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: BuildBroadPhase dispatch.");
                return false;
            }
            builtBroadPhase = true;
        }
        if (constants.staticBodyCount > 0u && mImpl->sceneState.staticBroadPhaseDirty())
        {
            mImpl->sceneState.setStaticBroadPhaseDirty(false);
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::BuildBroadPhase, builtBroadPhase);

        std::uint32_t pairCount = 0u;
        if (activeMovingCount > 0u)
        {
            if (!mImpl->passDispatcher.finalizeBroadPhasePairs(
                    computeBackend.computeContext, mImpl->sceneState, activeMovingCount, constants))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: FinalizePairs dispatch.");
                return false;
            }

            if (!mImpl->sceneState.readbackBroadPhaseMetaBlocking(computeBackend.computeContext,
                                                                  broadPhaseMeta))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: pair meta readback.");
                return false;
            }
            if (broadPhaseMeta.overflow != 0u)
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: candidate pair overflow (required=",
                                  broadPhaseMeta.requiredPairCount,
                                  ", capacity=", mImpl->sceneState.candidatePairCapacity(), ").");
                return false;
            }

            pairCount                    = broadPhaseMeta.candidatePairCount;
            constants.candidatePairCount = pairCount;
            if (!mImpl->passDispatcher.emitBroadPhasePairs(
                    computeBackend.computeContext, mImpl->sceneState, activeMovingCount, constants))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: typed pair emission dispatch.");
                return false;
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs, true);
        }
        else
        {
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs, false);
        }

        if (pairCount > 0u)
        {
            if (!mImpl->passDispatcher.generateContacts(computeBackend.computeContext,
                                                        mImpl->sceneState, pairCount))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: GenerateContacts dispatch.");
                return false;
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, true);

            if (!mImpl->passDispatcher.solveConstraints(computeBackend.computeContext,
                                                        mImpl->sceneState, rigidBodyCount,
                                                        pairCount, iterations, constants))
            {
                LOG_ERROR_MESSAGE("PhysicsSolver::step failed: SolveConstraints dispatch.");
                return false;
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, true);
        }
        else
        {
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts, false);
            markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, false);
        }

        constants.iterationIndex = iterations;
        if (!mImpl->passDispatcher.updateVelocities(computeBackend.computeContext,
                                                    mImpl->sceneState, rigidBodyCount, constants))
        {
            LOG_ERROR_MESSAGE("PhysicsSolver::step failed: UpdateVelocities dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateVelocities, true);

        if (substep + 1u < substeps && !mImpl->sceneState.copyPredictedRigidBodiesToPersistentState(
                                           computeBackend.computeContext, rigidBodyCount))
        {
            LOG_ERROR_MESSAGE(
                "PhysicsSolver::step failed: copyPredictedRigidBodiesToPersistentState.");
            return false;
        }
    }

    if (!mDesc.enableBlockingReadback)
    {
        markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, false);
        return true;
    }

    if (!mImpl->sceneState.readbackPredictedRigidStateBlocking(computeBackend.computeContext, world,
                                                               rigidBodyCount))
    {
        LOG_ERROR_MESSAGE("PhysicsSolver::step failed: readbackPredictedRigidStateBlocking.");
        return false;
    }

    markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults, true);
    return true;
}

const PhysicsSolverStageStats& PhysicsSolver::lastStageStats() const noexcept
{
    return mImpl->stageStats;
}

PhysicsGpuSceneView PhysicsSolver::gpuSceneView() const noexcept
{
    return mImpl != nullptr ? mImpl->sceneState.sceneView() : PhysicsGpuSceneView{};
}

} // namespace cressim::neo::physics
