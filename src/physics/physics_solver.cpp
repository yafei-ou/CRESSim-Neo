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

std::uint32_t resolveIterations(std::uint32_t overrideValue, std::uint32_t defaultValue)
{
    return overrideValue > 0u ? overrideValue : defaultValue;
}

std::uint32_t nextPowerOfTwo(std::uint32_t value)
{
    if (value <= 1u)
    {
        return 1u;
    }

    --value;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

} // namespace

struct PhysicsSolver::Impl
{
    PhysicsSceneGpuState sceneState;
    PhysicsPassDispatcher passDispatcher;
    bool lastStepHadRigidBroadPhaseWork = false;
    bool lastStepHadSoftPairWork        = false;
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

    if (!mDesc.enableGpuCompute)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    gpu::GpuGraphicsBackendContext graphicsBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        !mDevice.tryGetGraphicsBackendContext(graphicsBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr)
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: missing physics backend context.");
        return false;
    }

    world.ensureDerivedStateUpToDate();

    const std::uint32_t rigidBodyCount       = world.rigidBodyCount();
    const std::uint32_t colliderCount        = world.colliderCount();
    const SoftParticleSoAHost &softParticles = world.softParticles();
    const std::vector<SoftEdge> &softEdges   = world.softEdges();
    const std::vector<SoftTet> &softTets     = world.softTets();
    const SoftRenderDataHost &softRenderData = world.softRenderData();
    const std::uint32_t softParticleCount    = static_cast<std::uint32_t>(softParticles.size());
    const std::uint32_t softEdgeCount        = static_cast<std::uint32_t>(softEdges.size());
    const std::uint32_t softTetCount         = static_cast<std::uint32_t>(softTets.size());
    const std::uint32_t softRenderTriangleCount =
        static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size());
    std::uint32_t softBodyBoundsChunkCount = 0u;
    for (const Diligent::uint2 &range : softRenderData.softBodyParticleRanges)
    {
        softBodyBoundsChunkCount += (range.y + 64u - 1u) / 64u;
    }
    float particleGridCellSize = 0.0f;
    for (const float radius : softParticles.radii)
    {
        particleGridCellSize = std::max(particleGridCellSize, radius * 2.0f);
    }
    particleGridCellSize   = std::max(particleGridCellSize, 0.1f);
    const bool hasSoftData = softParticleCount > 0u || softEdgeCount > 0u || softTetCount > 0u;
    if (rigidBodyCount == 0u && !hasSoftData)
    {
        return true;
    }

    if (!mImpl->sceneState.ensureCapacity(
            computeBackend.renderDevice, rigidBodyCount, colliderCount, softParticleCount,
            softEdgeCount, softTetCount,
            static_cast<std::uint32_t>(softRenderData.fallbackNormals.size()),
            static_cast<std::uint32_t>(softRenderData.vertexTriangleIndices.size()),
            softRenderTriangleCount,
            static_cast<std::uint32_t>(softRenderData.softBodyParticleRanges.size()),
            softBodyBoundsChunkCount,
            gpu::contextMaskForId(computeBackend.contextId) |
                gpu::contextMaskForId(graphicsBackend.contextId),
            mDevice.supportsNativePhysicsFloatAtomics()))
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
    const std::uint32_t softContactIterations =
        resolveIterations(mDesc.softContactIterations, defaultIterations);
    const std::uint32_t rigidRigidContactIterations =
        resolveIterations(mDesc.rigidRigidContactIterations, defaultIterations);
    const std::uint32_t maxPhaseIterations = std::max(
        std::max(softInternalIterations, softContactIterations), rigidRigidContactIterations);
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
        softConstants.rigidColliderCount        = colliderCount;
        softConstants.particleGridCellSize      = particleGridCellSize;
        softConstants.softCandidatePairCapacity = mImpl->sceneState.softCandidatePairCapacity();
        softConstants.softCellRangeCapacity =
            nextPowerOfTwo(std::max<std::uint32_t>(softParticleCount * 2u, 1u));
        softConstants.softEdgeCount = softEdgeCount;
        softConstants.softTetCount  = softTetCount;

        const bool hasSoftPairWork = softParticleCount > 0u;
        const bool hasSoftInternalWork =
            softParticleCount > 0u && (softEdgeCount > 0u || softTetCount > 0u);
        const bool hasSoftSoftContactWork  = softParticleCount > 1u;
        const bool hasSoftRigidContactWork = softParticleCount > 0u && colliderCount > 0u;
        if (mImpl->sceneState.correctionBuffersNeedClear() &&
            !mImpl->passDispatcher.clearRigidCorrections(
                computeBackend.computeContext, mImpl->sceneState, rigidBodyCount, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearCorrections dispatch.");
            return false;
        }

        if (!mImpl->passDispatcher.predictRigid(computeBackend.computeContext, mImpl->sceneState,
                                                rigidBodyCount, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: PredictState dispatch.");
            return false;
        }
        const bool hasRigidBroadPhaseWork = world.activeMovingColliderCount() > 0u;
        constants.activeMovingCount       = world.activeMovingColliderCount();
        constants.staticBodyCount         = world.staticColliderCount();
        if (rigidBodyCount != 0u)
        {
            if (!mImpl->passDispatcher.updateRigidWorldAabbs(
                    computeBackend.computeContext, mImpl->sceneState, colliderCount, constants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateRigidWorldAabbs dispatch.");
                return false;
            }
            if (colliderCount > 0u &&
                !mImpl->passDispatcher.compactBroadPhaseBodySets(
                    computeBackend.computeContext, mImpl->sceneState, colliderCount, constants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildBroadPhase compaction dispatch.");
                return false;
            }
            if ((constants.activeMovingCount > 0u || constants.staticBodyCount > 0u) &&
                !mImpl->passDispatcher.buildBroadPhase(computeBackend.computeContext,
                                                       mImpl->sceneState,
                                                       constants.activeMovingCount, constants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: BuildBroadPhase dispatch.");
                return false;
            }
            if (constants.staticBodyCount > 0u && mImpl->sceneState.staticBroadPhaseDirty())
            {
                mImpl->sceneState.setStaticBroadPhaseDirty(false);
            }

            if (hasRigidBroadPhaseWork)
            {
                if (!mImpl->passDispatcher.finalizeBroadPhasePairs(
                        computeBackend.computeContext, mImpl->sceneState,
                        constants.activeMovingCount, constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: FinalizePairs dispatch.");
                    return false;
                }
                if (!mImpl->passDispatcher.emitBroadPhasePairs(
                        computeBackend.computeContext, mImpl->sceneState,
                        constants.activeMovingCount, constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: typed pair emission dispatch.");
                    return false;
                }
                if (!mImpl->passDispatcher.prepareRigidIndirectArgs(computeBackend.computeContext,
                                                                    mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: PrepareRigidIndirectArgs dispatch.");
                    return false;
                }
            }
        }

        if (!mImpl->passDispatcher.predictSoft(computeBackend.computeContext, mImpl->sceneState,
                                               softParticleCount, softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SoftPredict dispatch.");
            return false;
        }

        if (hasSoftSoftContactWork)
        {
            if (!mImpl->passDispatcher.buildParticleBroadPhaseEntries(
                    computeBackend.computeContext, mImpl->sceneState, softParticleCount,
                    softConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildParticleBroadPhaseEntries dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.buildParticleBroadPhaseKeys(
                    computeBackend.computeContext, mImpl->sceneState, softParticleCount,
                    softConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildParticleBroadPhaseKeys dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.sortParticleBroadPhase(computeBackend.computeContext,
                                                              mImpl->sceneState, softParticleCount))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SoftRigidRadixSort dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.clearParticleCellRanges(
                    computeBackend.computeContext, mImpl->sceneState,
                    softConstants.softCellRangeCapacity, softConstants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearParticleCellRanges dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.buildParticleCellRanges(computeBackend.computeContext,
                                                               mImpl->sceneState, softParticleCount,
                                                               softConstants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: BuildParticleCellRanges dispatch.");
                return false;
            }
        }

        mImpl->lastStepHadRigidBroadPhaseWork = hasRigidBroadPhaseWork;
        mImpl->lastStepHadSoftPairWork        = hasSoftPairWork;
        if (hasSoftPairWork)
        {
            if (!mImpl->passDispatcher.clearSoftNeighborMeta(computeBackend.computeContext,
                                                             mImpl->sceneState))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearSoftNeighborMeta dispatch.");
                return false;
            }
            if (hasSoftSoftContactWork && !mImpl->passDispatcher.buildSoftSoftCandidatePairs(
                                              computeBackend.computeContext, mImpl->sceneState,
                                              softParticleCount, softConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildSoftSoftCandidatePairs dispatch.");
                return false;
            }
            if (hasSoftRigidContactWork && !mImpl->passDispatcher.buildSoftRigidCandidatePairs(
                                               computeBackend.computeContext, mImpl->sceneState,
                                               softParticleCount, softConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildSoftRigidCandidatePairs dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.prepareSoftCandidateIndirectArgs(
                    computeBackend.computeContext, mImpl->sceneState))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: PrepareSoftCandidateIndirectArgs dispatch.");
                return false;
            }
        }

        const std::uint32_t softConstraintThreadCount =
            std::max(softParticleCount, std::max(softEdgeCount, softTetCount));
        if ((hasSoftInternalWork || hasSoftSoftContactWork || hasSoftRigidContactWork) &&
            !mImpl->passDispatcher.clearSoftConstraintState(
                computeBackend.computeContext, mImpl->sceneState, softConstraintThreadCount,
                softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearSoftConstraintState dispatch.");
            return false;
        }

        const bool hasAnyPositionSolveWork =
            (hasSoftInternalWork && softInternalIterations > 0u) ||
            (hasSoftSoftContactWork && softContactIterations > 0u) ||
            (hasSoftRigidContactWork && softContactIterations > 0u) ||
            (hasRigidBroadPhaseWork && rigidRigidContactIterations > 0u);
        if (hasAnyPositionSolveWork)
        {
            for (std::uint32_t iteration = 0u; iteration < maxPhaseIterations; ++iteration)
            {
                constants.iterationIndex = iteration;
                const bool runSoftInternal =
                    hasSoftInternalWork && iteration < softInternalIterations;
                const bool runSoftContacts =
                    hasSoftSoftContactWork && iteration < softContactIterations;
                const bool runSoftRigidContacts =
                    hasSoftRigidContactWork && iteration < softContactIterations;
                const bool runRigidRigidContacts =
                    hasRigidBroadPhaseWork && iteration < rigidRigidContactIterations;
                const bool needContactSoftApply = runSoftContacts || runSoftRigidContacts;
                const bool needRigidApply       = runSoftRigidContacts || runRigidRigidContacts;

                if (runSoftInternal && !mImpl->passDispatcher.solveSoftEdgeConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           softEdgeCount, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSoftEdgeConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal &&
                    !mImpl->passDispatcher.applySoftEdgeCorrections(
                        computeBackend.computeContext, mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplySoftEdgeCorrections dispatch.");
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
                if (runSoftInternal &&
                    !mImpl->passDispatcher.applySoftTetCorrections(
                        computeBackend.computeContext, mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplySoftTetCorrections dispatch.");
                    return false;
                }
                if (runSoftContacts &&
                    !mImpl->passDispatcher.generateSoftContacts(computeBackend.computeContext,
                                                                mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: GenerateSoftContacts dispatch.");
                    return false;
                }
                if (runSoftContacts &&
                    !mImpl->passDispatcher.compactSoftContacts(computeBackend.computeContext,
                                                               mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: CompactSoftContacts dispatch.");
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
                    !mImpl->passDispatcher.compactSoftRigidContacts(
                        computeBackend.computeContext, mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: CompactSoftRigidContacts dispatch.");
                    return false;
                }
                if ((runSoftContacts || runSoftRigidContacts) &&
                    !mImpl->passDispatcher.prepareSoftActiveIndirectArgs(
                        computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: PrepareSoftActiveIndirectArgs dispatch.");
                    return false;
                }
                if (runSoftContacts &&
                    !mImpl->passDispatcher.solveSoftContacts(computeBackend.computeContext,
                                                             mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SolveSoftContacts dispatch.");
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
                if (runRigidRigidContacts && !mImpl->passDispatcher.generateRigidContacts(
                                                 computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: GenerateRigidContacts dispatch.");
                    return false;
                }
                if (runRigidRigidContacts && !mImpl->passDispatcher.solveRigidContactConstraints(
                                                 computeBackend.computeContext, mImpl->sceneState,
                                                 rigidBodyCount, constants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveRigidContactConstraints dispatch.");
                    return false;
                }
                if (needContactSoftApply &&
                    !mImpl->passDispatcher.applySoftPositionCorrections(
                        computeBackend.computeContext, mImpl->sceneState, softConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplySoftPositionCorrections dispatch.");
                    return false;
                }
                if (needRigidApply && !mImpl->passDispatcher.applyRigidCorrections(
                                          computeBackend.computeContext, mImpl->sceneState,
                                          rigidBodyCount, constants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyRigidCorrections dispatch.");
                    return false;
                }
            }
        }
        constants.iterationIndex = maxPhaseIterations;
        if (!mImpl->passDispatcher.updateRigidVelocities(
                computeBackend.computeContext, mImpl->sceneState, rigidBodyCount, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateRigidVelocities dispatch.");
            return false;
        }

        if (!mImpl->passDispatcher.updateSoftVelocities(
                computeBackend.computeContext, mImpl->sceneState, softParticleCount, softConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateSoftVelocities dispatch.");
            return false;
        }
        if (hasSoftSoftContactWork && softContactIterations > 0u &&
            !mImpl->passDispatcher.solveSoftContactVelocities(
                computeBackend.computeContext, mImpl->sceneState, softParticleCount,
                softContactIterations))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SolveSoftContactVelocities dispatch.");
            return false;
        }
        if (hasSoftRigidContactWork && softContactIterations > 0u &&
            !mImpl->passDispatcher.solveSoftRigidContactVelocities(
                computeBackend.computeContext, mImpl->sceneState, softParticleCount, rigidBodyCount,
                softContactIterations, constants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: SolveSoftRigidContactVelocities dispatch.");
            return false;
        }
        if (!mImpl->passDispatcher.updateSoftTriangleNormals(
                computeBackend.computeContext, mImpl->sceneState, softRenderTriangleCount))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateSoftTriangleNormals dispatch.");
            return false;
        }
        if (!mImpl->passDispatcher.updateSoftRenderNormals(
                computeBackend.computeContext, mImpl->sceneState,
                static_cast<std::uint32_t>(softRenderData.fallbackNormals.size())))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateSoftRenderNormals dispatch.");
            return false;
        }
        if (!mImpl->passDispatcher.updateSoftBodyBounds(computeBackend.computeContext,
                                                        mImpl->sceneState, world.softBodyCount(),
                                                        softBodyBoundsChunkCount))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateSoftBodyBounds dispatch.");
            return false;
        }

        if (hasRigidBroadPhaseWork && rigidRigidContactIterations > 0u &&
            !mImpl->passDispatcher.solveRigidContactVelocities(
                computeBackend.computeContext, mImpl->sceneState, rigidBodyCount,
                rigidRigidContactIterations, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SolveRigidContactVelocities dispatch.");
            return false;
        }

        if (!mImpl->sceneState.copyPredictedRigidBodiesToPersistentState(
                computeBackend.computeContext, rigidBodyCount))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: copyPredictedRigidBodiesToPersistentState.");
            return false;
        }
    }

    if (!mDesc.enableBlockingReadback)
    {
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

    return true;
}

bool PhysicsSolver::validateGpuMetaBlocking()
{
    if (!mInitialized || !mDesc.enableGpuCompute)
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        CRESSIM_LOG_ERROR(
            "PhysicsSolver::validateGpuMetaBlocking failed: missing physics backend context.");
        return false;
    }

    if (mImpl->lastStepHadRigidBroadPhaseWork)
    {
        GpuBroadPhaseMeta broadPhaseMeta{};
        if (!mImpl->sceneState.readbackBroadPhaseMetaBlocking(computeBackend.computeContext,
                                                              broadPhaseMeta))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::validateGpuMetaBlocking failed: broad-phase meta readback.");
            return false;
        }
        if (broadPhaseMeta.overflow != 0u)
        {
            CRESSIM_LOG_ERROR("PhysicsSolver validation failed: candidate pair overflow (required=",
                              broadPhaseMeta.requiredPairCount,
                              ", capacity=", mImpl->sceneState.candidatePairCapacity(), ").");
            return false;
        }
    }

    if (mImpl->lastStepHadSoftPairWork)
    {
        GpuSoftNeighborMeta softNeighborMeta{};
        if (!mImpl->sceneState.readbackSoftNeighborMetaBlocking(computeBackend.computeContext,
                                                                softNeighborMeta))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::validateGpuMetaBlocking failed: soft neighbor meta readback.");
            return false;
        }
        if (softNeighborMeta.softSoftCandidateOverflow != 0u ||
            softNeighborMeta.softRigidCandidateOverflow != 0u)
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver validation failed: soft candidate overflow (soft-soft required=",
                softNeighborMeta.requiredSoftSoftCandidateCount,
                ", soft-rigid required=", softNeighborMeta.requiredSoftRigidCandidateCount,
                ", capacity=", mImpl->sceneState.softCandidatePairCapacity(), ").");
            return false;
        }
    }

    return true;
}

PhysicsGpuSceneView PhysicsSolver::gpuSceneView() const noexcept
{
    return mImpl != nullptr ? mImpl->sceneState.sceneView() : PhysicsGpuSceneView{};
}

} // namespace cressim::neo::physics
