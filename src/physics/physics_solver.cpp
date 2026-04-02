#include "physics/physics_solver.h"

#include "common/logger.h"
#include "physics/physics_pass_dispatcher.h"
#include "physics/physics_scene_gpu_state.h"

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

void markStage(PhysicsSolverStageStats &stats, PhysicsSolverStage stage, bool executed)
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

void markAllStagesSkipped(PhysicsSolverStageStats &stats)
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

std::uint32_t resolveIterations(std::uint32_t overrideValue, std::uint32_t defaultValue)
{
    return overrideValue > 0u ? overrideValue : defaultValue;
}

} // namespace

struct PhysicsSolver::Impl
{
    PhysicsSceneGpuState sceneState;
    PhysicsPassDispatcher passDispatcher;
    PhysicsSolverStageStats stageStats{};
};

PhysicsSolver::PhysicsSolver(gpu::GpuDevice &device, const PhysicsSolverDesc &desc)
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
        CRESSIM_LOG_ERROR("PhysicsSolver: failed to get physics GPU context.");
        return false;
    }

    mImpl = std::make_unique<Impl>();
    if (!mImpl->passDispatcher.initialize(mDevice, computeContext.contextId))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver: failed to initialize physics pass dispatcher.");
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

bool PhysicsSolver::step(const common::FrameContext &frameContext, PhysicsWorld &world)
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
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: missing physics backend context.");
        return false;
    }

    world.ensureDerivedStateUpToDate();

    const std::uint32_t rigidBodyCount = world.rigidBodyCount();
    const std::uint32_t colliderCount  = world.colliderCount();
    const std::uint32_t softParticleCount =
        static_cast<std::uint32_t>(world.softParticles().size());
    const std::uint32_t softEdgeCount = static_cast<std::uint32_t>(world.softEdges().size());
    const std::uint32_t softTetCount  = static_cast<std::uint32_t>(world.softTets().size());
    const std::uint32_t rigidSurfaceParticleCount =
        static_cast<std::uint32_t>(world.rigidSurfaceParticles().size());
    float particleGridCellSize = 0.0f;
    for (const float radius : world.softParticles().radii)
    {
        particleGridCellSize = std::max(particleGridCellSize, radius * 2.0f);
    }
    for (const float radius : world.rigidSurfaceParticles().sampleRadii)
    {
        particleGridCellSize = std::max(particleGridCellSize, radius * 2.0f);
    }
    particleGridCellSize   = std::max(particleGridCellSize, 0.1f);
    const bool hasSoftData = softParticleCount > 0u || softEdgeCount > 0u || softTetCount > 0u ||
                             rigidSurfaceParticleCount > 0u;
    if (rigidBodyCount == 0u && !hasSoftData)
    {
        markAllStagesSkipped(mImpl->stageStats);
        return true;
    }

    if (!mImpl->sceneState.ensureCapacity(
            computeBackend.renderDevice, rigidBodyCount, colliderCount, softParticleCount,
            softEdgeCount, softTetCount, rigidSurfaceParticleCount, computeBackend.contextId))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ensureCapacity.");
        return false;
    }
    if (!mImpl->sceneState.uploadWorldState(computeBackend.computeContext, world, rigidBodyCount,
                                            colliderCount))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: uploadWorldState.");
        return false;
    }

    const std::uint32_t substeps          = std::max<std::uint32_t>(mDesc.substeps, 1u);
    const std::uint32_t defaultIterations = std::max<std::uint32_t>(mDesc.defaultIterations, 1u);
    const std::uint32_t softInternalIterations =
        resolveIterations(mDesc.softInternalIterations, defaultIterations);
    const std::uint32_t softRigidContactIterations =
        resolveIterations(mDesc.softRigidContactIterations, defaultIterations);
    const std::uint32_t rigidRigidContactIterations =
        resolveIterations(mDesc.rigidRigidContactIterations, defaultIterations);
    const std::uint32_t maxPhaseIterations = std::max(
        std::max(softInternalIterations, softRigidContactIterations), rigidRigidContactIterations);
    const float substepDt = frameContext.deltaSeconds / static_cast<float>(substeps);

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        GpuRigidDispatchConstants constants{};
        constants.dt                    = substepDt;
        constants.rigidBodyCount        = rigidBodyCount;
        constants.colliderCount         = colliderCount;
        constants.candidatePairCapacity = mImpl->sceneState.candidatePairCapacity();
        constants.substepIndex          = substep;
        constants.solverIterations      = maxPhaseIterations;
        GpuSoftDispatchConstants softConstants{};
        softConstants.dt                        = substepDt;
        softConstants.softParticleCount         = softParticleCount;
        softConstants.rigidSurfaceParticleCount = rigidSurfaceParticleCount;
        softConstants.particleGridCellSize      = particleGridCellSize;
        softConstants.softRigidCandidatePairCapacity =
            mImpl->sceneState.softRigidCandidatePairCapacity();
        softConstants.softEdgeCount = softEdgeCount;
        softConstants.softTetCount  = softTetCount;

        bool executedPredictStage        = false;
        const bool hasSoftBroadPhaseWork = (softParticleCount + rigidSurfaceParticleCount) > 0u;
        const bool hasSoftPairWork       = softParticleCount > 0u;
        const bool hasSoftInternalWork =
            softParticleCount > 0u && (softEdgeCount > 0u || softTetCount > 0u);
        const bool hasSoftRigidContactWork =
            softParticleCount > 0u && rigidSurfaceParticleCount > 0u;

        if (mImpl->sceneState.correctionBuffersNeedClear() &&
            !mImpl->passDispatcher.clearCorrections(computeBackend.computeContext,
                                                    mImpl->sceneState, rigidBodyCount, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearCorrections dispatch.");
            return false;
        }

        if (!mImpl->passDispatcher.predict(computeBackend.computeContext, mImpl->sceneState,
                                           rigidBodyCount, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: PredictState dispatch.");
            return false;
        }
        executedPredictStage = executedPredictStage || rigidBodyCount > 0u;

        if (!mImpl->passDispatcher.predictSoft(computeBackend.computeContext, mImpl->sceneState,
                                               softParticleCount, softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SoftPredict dispatch.");
            return false;
        }
        executedPredictStage = executedPredictStage || softParticleCount > 0u;

        if (!mImpl->passDispatcher.updateRigidSurfaceWorldPositions(
                computeBackend.computeContext, mImpl->sceneState, rigidSurfaceParticleCount,
                softConstants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: UpdateRigidSurfaceWorldPositions dispatch.");
            return false;
        }
        markStage(mImpl->stageStats, PhysicsSolverStage::PredictState, executedPredictStage);

        if (!mImpl->passDispatcher.buildSoftRigidBroadPhaseParticles(
                computeBackend.computeContext, mImpl->sceneState,
                softParticleCount + rigidSurfaceParticleCount, softConstants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: BuildSoftRigidBroadPhaseParticles dispatch.");
            return false;
        }
        if (!mImpl->passDispatcher.buildSoftRigidBroadPhaseKeys(
                computeBackend.computeContext, mImpl->sceneState,
                softParticleCount + rigidSurfaceParticleCount, softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: BuildSoftRigidBroadPhaseKeys dispatch.");
            return false;
        }
        if (!mImpl->passDispatcher.sortSoftRigidBroadPhase(
                computeBackend.computeContext, mImpl->sceneState,
                softParticleCount + rigidSurfaceParticleCount))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SoftRigidRadixSort dispatch.");
            return false;
        }

        if (!mImpl->passDispatcher.emitSoftRigidCandidatePairs(
                computeBackend.computeContext, mImpl->sceneState, softParticleCount, softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: EmitSoftRigidCandidatePairs dispatch.");
            return false;
        }

        const std::uint32_t softConstraintThreadCount =
            std::max(softParticleCount, std::max(softEdgeCount, softTetCount));
        if ((hasSoftInternalWork || hasSoftRigidContactWork) &&
            !mImpl->passDispatcher.clearSoftConstraintState(
                computeBackend.computeContext, mImpl->sceneState, softConstraintThreadCount,
                softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearSoftConstraintState dispatch.");
            return false;
        }

        std::uint32_t pairCount = 0u;
        bool builtBroadPhase    = hasSoftBroadPhaseWork;
        if (rigidBodyCount == 0u)
        {
            markStage(mImpl->stageStats, PhysicsSolverStage::UpdateWorldAabbs, false);
            markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs,
                      hasSoftPairWork);
        }
        else
        {
            if (!mImpl->passDispatcher.updateWorldAabbs(
                    computeBackend.computeContext, mImpl->sceneState, colliderCount, constants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateWorldAabbs dispatch.");
                return false;
            }
            markStage(mImpl->stageStats, PhysicsSolverStage::UpdateWorldAabbs, true);

            if (colliderCount > 0u &&
                !mImpl->passDispatcher.compactBroadPhaseBodySets(
                    computeBackend.computeContext, mImpl->sceneState, colliderCount, constants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildBroadPhase compaction dispatch.");
                return false;
            }

            GpuBroadPhaseMeta broadPhaseMeta{};
            if (colliderCount > 0u && !mImpl->sceneState.readbackBroadPhaseMetaBlocking(
                                          computeBackend.computeContext, broadPhaseMeta))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: readbackBroadPhaseMetaBlocking.");
                return false;
            }

            const std::uint32_t activeMovingCount =
                colliderCount > 0u ? broadPhaseMeta.activeMovingCount : 0u;
            constants.activeMovingCount = activeMovingCount;
            constants.staticBodyCount   = colliderCount > 0u ? broadPhaseMeta.staticBodyCount : 0u;
            if (activeMovingCount > 0u)
            {
                if (!mImpl->passDispatcher.buildBroadPhase(computeBackend.computeContext,
                                                           mImpl->sceneState, activeMovingCount,
                                                           constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: BuildBroadPhase dispatch.");
                    return false;
                }
                builtBroadPhase = true;
            }
            if (constants.staticBodyCount > 0u && mImpl->sceneState.staticBroadPhaseDirty())
            {
                mImpl->sceneState.setStaticBroadPhaseDirty(false);
            }

            if (activeMovingCount > 0u)
            {
                if (!mImpl->passDispatcher.finalizeBroadPhasePairs(computeBackend.computeContext,
                                                                   mImpl->sceneState,
                                                                   activeMovingCount, constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: FinalizePairs dispatch.");
                    return false;
                }

                if (!mImpl->sceneState.readbackBroadPhaseMetaBlocking(computeBackend.computeContext,
                                                                      broadPhaseMeta))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: pair meta readback.");
                    return false;
                }
                if (broadPhaseMeta.overflow != 0u)
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: candidate pair overflow (required=",
                        broadPhaseMeta.requiredPairCount,
                        ", capacity=", mImpl->sceneState.candidatePairCapacity(), ").");
                    return false;
                }

                pairCount                    = broadPhaseMeta.candidatePairCount;
                constants.candidatePairCount = pairCount;
                if (!mImpl->passDispatcher.emitBroadPhasePairs(computeBackend.computeContext,
                                                               mImpl->sceneState, activeMovingCount,
                                                               constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: typed pair emission dispatch.");
                    return false;
                }
                markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs, true);
            }
            else
            {
                markStage(mImpl->stageStats, PhysicsSolverStage::GenerateBroadPhasePairs,
                          hasSoftPairWork);
            }
        }

        markStage(mImpl->stageStats, PhysicsSolverStage::BuildBroadPhase, builtBroadPhase);

        const bool hasAnyPositionSolveWork =
            (hasSoftInternalWork && softInternalIterations > 0u) ||
            (hasSoftRigidContactWork && softRigidContactIterations > 0u) ||
            (pairCount > 0u && rigidRigidContactIterations > 0u);
        if (hasAnyPositionSolveWork)
        {
            for (std::uint32_t iteration = 0u; iteration < maxPhaseIterations; ++iteration)
            {
                constants.iterationIndex = iteration;
                const bool runSoftInternal =
                    hasSoftInternalWork && iteration < softInternalIterations;
                const bool runSoftRigidContacts =
                    hasSoftRigidContactWork && iteration < softRigidContactIterations;
                const bool runRigidRigidContacts =
                    pairCount > 0u && iteration < rigidRigidContactIterations;
                const bool needSoftApply  = runSoftInternal || runSoftRigidContacts;
                const bool needRigidApply = runSoftRigidContacts || runRigidRigidContacts;

                if (runSoftInternal && !mImpl->passDispatcher.solveSoftEdgeConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           softEdgeCount, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSoftEdgeConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.solveSoftTetConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           softTetCount, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSoftTetConstraints dispatch.");
                    return false;
                }
                if (runSoftRigidContacts &&
                    !mImpl->passDispatcher.generateSoftRigidContacts(
                        computeBackend.computeContext, mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: GenerateSoftRigidContacts dispatch.");
                    return false;
                }
                if (runSoftRigidContacts &&
                    !mImpl->passDispatcher.solveSoftRigidContacts(computeBackend.computeContext,
                                                                  mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSoftRigidContacts dispatch.");
                    return false;
                }
                if (runRigidRigidContacts &&
                    !mImpl->passDispatcher.generateContacts(computeBackend.computeContext,
                                                            mImpl->sceneState, pairCount))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: GenerateContacts dispatch.");
                    return false;
                }
                if (runRigidRigidContacts && !mImpl->passDispatcher.solveConstraints(
                                                 computeBackend.computeContext, mImpl->sceneState,
                                                 rigidBodyCount, pairCount, constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SolveConstraints dispatch.");
                    return false;
                }
                if (needSoftApply &&
                    !mImpl->passDispatcher.applySoftPositionCorrections(
                        computeBackend.computeContext, mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplySoftPositionCorrections dispatch.");
                    return false;
                }
                if (needRigidApply && !mImpl->passDispatcher.applyCorrections(
                                          computeBackend.computeContext, mImpl->sceneState,
                                          rigidBodyCount, constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ApplyCorrections dispatch.");
                    return false;
                }
            }
        }

        markStage(mImpl->stageStats, PhysicsSolverStage::GenerateContacts,
                  (hasSoftRigidContactWork && softRigidContactIterations > 0u) ||
                      (pairCount > 0u && rigidRigidContactIterations > 0u));
        markStage(mImpl->stageStats, PhysicsSolverStage::SolveConstraints, hasAnyPositionSolveWork);

        constants.iterationIndex = maxPhaseIterations;
        if (!mImpl->passDispatcher.updateVelocities(computeBackend.computeContext,
                                                    mImpl->sceneState, rigidBodyCount, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateVelocities dispatch.");
            return false;
        }
        bool executedVelocityStage = rigidBodyCount > 0u;

        if (!mImpl->passDispatcher.updateSoftVelocities(
                computeBackend.computeContext, mImpl->sceneState, softParticleCount, softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateSoftVelocities dispatch.");
            return false;
        }
        executedVelocityStage = executedVelocityStage || softParticleCount > 0u;
        markStage(mImpl->stageStats, PhysicsSolverStage::UpdateVelocities, executedVelocityStage);

        if (pairCount > 0u && rigidRigidContactIterations > 0u &&
            !mImpl->passDispatcher.solveContactVelocities(
                computeBackend.computeContext, mImpl->sceneState, rigidBodyCount, pairCount,
                rigidRigidContactIterations, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SolveContactVelocities dispatch.");
            return false;
        }

        if (substep + 1u < substeps && !mImpl->sceneState.copyPredictedRigidBodiesToPersistentState(
                                           computeBackend.computeContext, rigidBodyCount))
        {
            CRESSIM_LOG_ERROR(
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
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: readbackPredictedRigidStateBlocking.");
        return false;
    }
    if (!mImpl->sceneState.readbackPredictedSoftStateBlocking(computeBackend.computeContext, world,
                                                              softParticleCount))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: readbackPredictedSoftStateBlocking.");
        return false;
    }

    markStage(mImpl->stageStats, PhysicsSolverStage::CommitResults,
              rigidBodyCount > 0u || softParticleCount > 0u);
    return true;
}

const PhysicsSolverStageStats &PhysicsSolver::lastStageStats() const noexcept
{
    return mImpl->stageStats;
}

PhysicsGpuSceneView PhysicsSolver::gpuSceneView() const noexcept
{
    return mImpl != nullptr ? mImpl->sceneState.sceneView() : PhysicsGpuSceneView{};
}

} // namespace cressim::neo::physics
