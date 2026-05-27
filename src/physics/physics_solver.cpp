#include "physics/physics_solver.h"

#include "common/logger.h"
#include "gpu/cuda_interop.h"
#include "physics/physics_pass_dispatcher.h"
#include "physics/physics_scene_gpu_state.h"

#include <algorithm>
#include <array>
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

std::uint32_t buildUniqueQueueFamilyIndices(
    const Diligent::IDeviceContext *firstContext, const Diligent::IDeviceContext *secondContext,
    std::array<std::uint32_t, 2> &outQueueFamilyIndices)
{
    std::uint32_t count = 0u;
    const auto appendQueueId = [&outQueueFamilyIndices, &count](const Diligent::IDeviceContext *ctx)
    {
        if (ctx == nullptr)
        {
            return;
        }

        const std::uint32_t queueId = ctx->GetDesc().QueueId;
        for (std::uint32_t i = 0u; i < count; ++i)
        {
            if (outQueueFamilyIndices[i] == queueId)
            {
                return;
            }
        }

        if (count < outQueueFamilyIndices.size())
        {
            outQueueFamilyIndices[count++] = queueId;
        }
    };

    appendQueueId(firstContext);
    appendQueueId(secondContext);
    return count;
}

bool hasPhysicsGpuBackend(gpu::GpuDevice &device)
{
    gpu::GpuComputeBackendContext computeContext{};
    gpu::GpuGraphicsBackendContext graphicsContext{};
    return device.tryGetPhysicsBackendContext(computeContext) &&
           device.tryGetGraphicsBackendContext(graphicsContext) &&
           computeContext.renderDevice != nullptr && computeContext.computeContext != nullptr &&
           graphicsContext.renderDevice != nullptr && graphicsContext.graphicsContext != nullptr;
}

struct SoftPositionsCudaInteropProbe
{
    gpu::CudaSharedBufferBridge bridge;
    bool disabled                           = false;
    bool loggedSuccess                      = false;
};

bool ensureSoftPositionsCudaInteropProbe(SoftPositionsCudaInteropProbe &probe,
                                         Diligent::IRenderDevice *renderDevice)
{
    if (probe.disabled)
    {
        return false;
    }

    if (!gpu::CudaStream::supportsCudaInteropBuild() ||
        !gpu::CudaSharedBufferBridge::supportsCudaInteropBuild())
    {
        probe.disabled = true;
        return false;
    }

    if (!probe.bridge.isInitialized() &&
        !probe.bridge.initializeForVulkan(renderDevice,
                                          "CRESSimNeo.Physics.SoftPositionsCudaProbe"))
    {
        CRESSIM_LOG_WARNING("PhysicsSolver: failed to initialize CUDA interop bridge for soft"
                            " positions probe. Disabling probe.");
        probe.disabled = true;
        return false;
    }

    return true;
}

bool ensureSoftPositionsCudaInteropBuffer(
    SoftPositionsCudaInteropProbe &probe, const gpu::SharedExportBuffer &sharedBuffer)
{
    if (probe.disabled || !sharedBuffer.usesNativeSharedAllocation())
    {
        return false;
    }

    if (!probe.bridge.bindSharedBuffer(sharedBuffer))
    {
        CRESSIM_LOG_WARNING("PhysicsSolver: failed to import shared soft positions buffer into"
                            " CUDA. Disabling probe.");
        probe.disabled = true;
        return false;
    }
    return true;
}

bool runSoftPositionsCudaInteropProbe(SoftPositionsCudaInteropProbe &probe,
                                      Diligent::IRenderDevice *renderDevice,
                                      Diligent::IDeviceContext *computeContext,
                                      const gpu::SharedExportBuffer &sharedBuffer,
                                      const common::FrameContext &frameContext)
{
    if (!ensureSoftPositionsCudaInteropProbe(probe, renderDevice) ||
        !ensureSoftPositionsCudaInteropBuffer(probe, sharedBuffer) ||
        computeContext == nullptr)
    {
        return false;
    }

    if (!probe.bridge.synchronizeFromDeviceContext(computeContext))
    {
        CRESSIM_LOG_WARNING("PhysicsSolver: failed to synchronize CUDA soft positions probe."
                            " Disabling probe.");
        probe.disabled = true;
        return false;
    }

    std::array<float, 4> samplePosition = {0.0f, 0.0f, 0.0f, 0.0f};
    if (!probe.bridge.isInitialized() ||
        !probe.bridge.devicePointer() ||
        !probe.bridge.streamHandle())
    {
        probe.disabled = true;
        return false;
    }

    if (!probe.bridge.copyDeviceToHostAsync(samplePosition.data(), probe.bridge.devicePointer(),
                                            sizeof(samplePosition)) ||
        !probe.bridge.synchronizeStream())
    {
        CRESSIM_LOG_WARNING("PhysicsSolver: failed to read back CUDA soft positions probe sample."
                            " Disabling probe.");
        probe.disabled = true;
        return false;
    }

    if (!probe.loggedSuccess)
    {
        CRESSIM_LOG_INFO("PhysicsSolver: CUDA interop probe read shared soft position sample at"
                         " frame ", frameContext.frameIndex, ": (",
                         samplePosition[0], ", ", samplePosition[1], ", ",
                         samplePosition[2], ", ", samplePosition[3], ").");
        probe.loggedSuccess = true;
    }

    return true;
}

} // namespace

struct PhysicsSolver::Impl
{
    PhysicsSceneGpuState sceneState;
    PhysicsPassDispatcher passDispatcher;
    SoftPositionsCudaInteropProbe softPositionsCudaInteropProbe;
    bool lastStepHadRigidBroadPhaseWork             = false;
    bool lastStepHadSoftPairWork                    = false;
    std::uint64_t lastAppliedRigidBindingGeneration = 0u;
    std::uint64_t lastAppliedSoftBindingGeneration  = 0u;
};

PhysicsSolver::PhysicsSolver(gpu::GpuDevice &device, const PhysicsSolverDesc &desc)
    : mDevice(device), mDesc(desc), mImpl(std::make_unique<Impl>())
{
}

PhysicsSolver::~PhysicsSolver() = default;

bool PhysicsSolver::initialize()
{
    shutdown();

    if (!hasPhysicsGpuBackend(mDevice))
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

    if (!hasPhysicsGpuBackend(mDevice))
    {
        return true;
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

    const std::uint32_t rigidBodyCount                  = world.rigidBodyCount();
    const std::uint32_t colliderCount                   = world.colliderCount();
    const ParticleSoAHost &particles                    = world.particles();
    const std::vector<FluidMaterialGpu> &fluidMaterials = world.fluidMaterials();
    const std::vector<SoftEdge> &softEdges              = world.softEdges();
    const std::vector<SoftTet> &softTets                = world.softTets();
    const SoftRenderDataHost &softRenderData            = world.softRenderData();
    const RigidJointSceneHost &rigidJoints              = world.rigidJointScene();
    const std::uint32_t fluidCount                      = world.fluidCount();
    const std::uint32_t particleCount    = static_cast<std::uint32_t>(particles.size());
    const std::uint32_t softEdgeCount    = static_cast<std::uint32_t>(softEdges.size());
    const std::uint32_t softTetCount     = static_cast<std::uint32_t>(softTets.size());
    const std::uint32_t ballJointCount   = static_cast<std::uint32_t>(rigidJoints.ball.size());
    const std::uint32_t hingeJointCount  = static_cast<std::uint32_t>(rigidJoints.hinge.size());
    const std::uint32_t sliderJointCount = static_cast<std::uint32_t>(rigidJoints.slider.size());
    const std::uint32_t softRenderTriangleCount =
        static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size());
    const std::uint32_t softBodyBoundsChunkCount = world.softBodyBoundsChunkCount();
    const float particleGridCellSize             = world.particleGridCellSize();
    std::array<std::uint32_t, 2> sharedQueueFamilyIndices{};
    const std::uint32_t sharedQueueFamilyIndexCount = buildUniqueQueueFamilyIndices(
        computeBackend.computeContext, graphicsBackend.graphicsContext, sharedQueueFamilyIndices);
    const bool hasSoftData = particleCount > 0u || softEdgeCount > 0u || softTetCount > 0u;
    if (rigidBodyCount == 0u && !hasSoftData)
    {
        return true;
    }

    if (!mImpl->sceneState.ensureCapacity(
            computeBackend.renderDevice, rigidBodyCount, colliderCount, particleCount, fluidCount,
            static_cast<std::uint32_t>(world.particleContactMaterials().size()),
            static_cast<std::uint32_t>(fluidMaterials.size()), softEdgeCount, softTetCount,
            ballJointCount, hingeJointCount, sliderJointCount,
            static_cast<std::uint32_t>(softRenderData.fallbackNormals.size()),
            static_cast<std::uint32_t>(softRenderData.vertexTriangleIndices.size()),
            softRenderTriangleCount,
            static_cast<std::uint32_t>(softRenderData.softBodyParticleRanges.size()),
            softBodyBoundsChunkCount,
            gpu::contextMaskForId(computeBackend.contextId) |
                gpu::contextMaskForId(graphicsBackend.contextId),
            sharedQueueFamilyIndices.data(), sharedQueueFamilyIndexCount,
            mDevice.supportsNativePhysicsFloatAtomics()))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ensureCapacity.");
        return false;
    }

    const bool rigidBindingsChanged =
        mImpl->lastAppliedRigidBindingGeneration != mImpl->sceneState.rigidBindingGeneration();
    const bool softBindingsChanged =
        mImpl->lastAppliedSoftBindingGeneration != mImpl->sceneState.softBindingGeneration();
    if ((rigidBindingsChanged || softBindingsChanged) &&
        !mImpl->passDispatcher.recreateSceneBindingVariants())
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: recreateSceneBindingVariants.");
        return false;
    }
    mImpl->lastAppliedRigidBindingGeneration = mImpl->sceneState.rigidBindingGeneration();
    mImpl->lastAppliedSoftBindingGeneration  = mImpl->sceneState.softBindingGeneration();

    if (!mImpl->sceneState.uploadWorldState(computeBackend.computeContext, world, rigidBodyCount,
                                            colliderCount))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: uploadWorldState.");
        return false;
    }

    const std::uint32_t substeps          = std::max<std::uint32_t>(mDesc.substeps, 1u);
    const std::uint32_t defaultIterations = std::max<std::uint32_t>(mDesc.defaultIterations, 1u);
    const std::uint32_t fluidIterations =
        resolveIterations(mDesc.fluidIterations, defaultIterations);
    const std::uint32_t softInternalIterations =
        resolveIterations(mDesc.softInternalIterations, defaultIterations);
    const std::uint32_t softContactIterations =
        resolveIterations(mDesc.softContactIterations, defaultIterations);
    const std::uint32_t rigidJointIterations =
        resolveIterations(mDesc.rigidJointIterations, defaultIterations);
    const std::uint32_t rigidContactIterations =
        resolveIterations(mDesc.rigidRigidContactIterations, defaultIterations);
    const std::uint32_t maxPositionPhaseIterations =
        std::max(std::max(std::max(fluidIterations, softInternalIterations), softContactIterations),
                 std::max(rigidJointIterations, rigidContactIterations));
    const float substepDt = frameContext.deltaSeconds / static_cast<float>(substeps);

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        GpuRigidDispatchConstants constants{};
        constants.dt                    = substepDt;
        constants.rigidBodyCount        = rigidBodyCount;
        constants.colliderCount         = colliderCount;
        constants.candidatePairCapacity = mImpl->sceneState.candidatePairCapacity();
        GpuParticleDispatchConstants particleConstants{};
        particleConstants.dt                   = substepDt;
        particleConstants.particleCount        = particleCount;
        particleConstants.rigidColliderCount   = colliderCount;
        particleConstants.particleGridCellSize = particleGridCellSize;
        particleConstants.particleCandidatePairCapacity =
            mImpl->sceneState.particleCandidatePairCapacity();
        particleConstants.fluidBoundaryCandidatePairCapacity =
            mImpl->sceneState.fluidBoundaryCandidatePairCapacity();
        particleConstants.fluidNeighborPairCapacity = mImpl->sceneState.fluidNeighborPairCapacity();
        particleConstants.maxFluidNeighborhood      = mImpl->sceneState.maxFluidNeighborhood();
        particleConstants.particleCellRangeCapacity =
            nextPowerOfTwo(std::max<std::uint32_t>(particleCount * 2u, 1u));
        particleConstants.softEdgeCount   = softEdgeCount;
        particleConstants.softTetCount    = softTetCount;
        particleConstants.fluidIterations = fluidIterations;

        const bool hasParticleNeighborWork = particleCount > 0u;
        const bool hasFluidWork            = fluidCount > 0u && particleCount > 0u;
        const bool hasFluidBoundaryWork    = hasFluidWork && colliderCount > 0u;
        const bool hasSoftInternalWork =
            particleCount > 0u && (softEdgeCount > 0u || softTetCount > 0u);
        const bool hasSoftContactSolveWork = softContactIterations > 0u;
        const bool hasSoftSoftContactWork  = hasSoftContactSolveWork && particleCount > 1u;
        const bool hasSoftRigidContactWork =
            hasSoftContactSolveWork && particleCount > 0u && colliderCount > 0u;
        const bool hasParticleBroadPhaseWork = hasSoftSoftContactWork || hasFluidWork;
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
        const bool useInitialRigidContactSolve =
            hasRigidBroadPhaseWork && rigidContactIterations > 0u;
        constants.activeMovingCount = world.activeMovingColliderCount();
        constants.staticBodyCount   = world.staticColliderCount();
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
                if (useInitialRigidContactSolve &&
                    !mImpl->passDispatcher.generateRigidContacts(computeBackend.computeContext,
                                                                 mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: GenerateRigidContacts initial dispatch.");
                    return false;
                }
                if (useInitialRigidContactSolve &&
                    !mImpl->passDispatcher.initRigidContactVelocities(computeBackend.computeContext,
                                                                      mImpl->sceneState, constants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: InitRigidContactVelocities initial dispatch.");
                    return false;
                }
            }
        }

        if (!mImpl->passDispatcher.predictSoft(computeBackend.computeContext, mImpl->sceneState,
                                               particleCount, particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SoftPredict dispatch.");
            return false;
        }

        if (hasParticleBroadPhaseWork)
        {
            if (!mImpl->passDispatcher.buildParticleBroadPhaseEntries(
                    computeBackend.computeContext, mImpl->sceneState, particleCount,
                    particleConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildParticleBroadPhaseEntries dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.buildParticleBroadPhaseKeys(computeBackend.computeContext,
                                                                   mImpl->sceneState, particleCount,
                                                                   particleConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildParticleBroadPhaseKeys dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.sortParticleBroadPhase(computeBackend.computeContext,
                                                              mImpl->sceneState, particleCount))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SoftRigidRadixSort dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.clearParticleCellRanges(
                    computeBackend.computeContext, mImpl->sceneState,
                    particleConstants.particleCellRangeCapacity, particleConstants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearParticleCellRanges dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.buildParticleCellRanges(computeBackend.computeContext,
                                                               mImpl->sceneState, particleCount,
                                                               particleConstants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: BuildParticleCellRanges dispatch.");
                return false;
            }
        }

        mImpl->lastStepHadRigidBroadPhaseWork = hasRigidBroadPhaseWork;
        mImpl->lastStepHadSoftPairWork        = hasParticleNeighborWork;
        if (hasParticleNeighborWork)
        {
            if (!mImpl->passDispatcher.clearParticleNeighborMeta(computeBackend.computeContext,
                                                                 mImpl->sceneState))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearSoftNeighborMeta dispatch.");
                return false;
            }
            if (hasSoftSoftContactWork &&
                !mImpl->passDispatcher.buildParticleParticleCandidatePairs(
                    computeBackend.computeContext, mImpl->sceneState, particleCount,
                    particleConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildSoftSoftCandidatePairs dispatch.");
                return false;
            }
            if (hasSoftRigidContactWork && !mImpl->passDispatcher.buildParticleRigidCandidatePairs(
                                               computeBackend.computeContext, mImpl->sceneState,
                                               particleCount, particleConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildSoftRigidCandidatePairs dispatch.");
                return false;
            }
            if (hasFluidBoundaryWork && !mImpl->passDispatcher.buildFluidBoundaryCandidatePairs(
                                            computeBackend.computeContext, mImpl->sceneState,
                                            particleCount, particleConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: BuildFluidBoundaryCandidatePairs dispatch.");
                return false;
            }
            if ((hasSoftSoftContactWork || hasSoftRigidContactWork) &&
                !mImpl->passDispatcher.prepareParticleCandidateIndirectArgs(
                    computeBackend.computeContext, mImpl->sceneState))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: PrepareParticleCandidateIndirectArgs dispatch.");
                return false;
            }
        }

        const std::uint32_t softConstraintThreadCount =
            std::max(particleCount, std::max(softEdgeCount, softTetCount));
        if ((hasSoftInternalWork || hasSoftSoftContactWork || hasSoftRigidContactWork) &&
            !mImpl->passDispatcher.clearSoftConstraintState(
                computeBackend.computeContext, mImpl->sceneState, softConstraintThreadCount,
                particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearSoftConstraintState dispatch.");
            return false;
        }
        if (sliderJointCount > 0u &&
            !mImpl->passDispatcher.clearSliderJointConstraintState(
                computeBackend.computeContext, mImpl->sceneState, sliderJointCount))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ClearSliderJointConstraintState dispatch.");
            return false;
        }
        if (hingeJointCount > 0u &&
            !mImpl->passDispatcher.clearHingeJointConstraintState(
                computeBackend.computeContext, mImpl->sceneState, hingeJointCount))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ClearHingeJointConstraintState dispatch.");
            return false;
        }

        const bool hasAnyPositionSolveWork =
            hasFluidWork || (hasSoftInternalWork && softInternalIterations > 0u) ||
            (hasSoftSoftContactWork && softContactIterations > 0u) ||
            (hasSoftRigidContactWork && softContactIterations > 0u) ||
            useInitialRigidContactSolve ||
            ((ballJointCount > 0u || hingeJointCount > 0u || sliderJointCount > 0u) &&
             rigidJointIterations > 0u);
        if (hasAnyPositionSolveWork)
        {
            for (std::uint32_t iteration = 0u; iteration < maxPositionPhaseIterations; ++iteration)
            {
                particleConstants.iterationIndex = iteration;
                const bool runSoftInternal =
                    hasSoftInternalWork && iteration < softInternalIterations;
                const bool runSoftContacts =
                    hasSoftSoftContactWork && iteration < softContactIterations;
                const bool runSoftRigidContacts =
                    hasSoftRigidContactWork && iteration < softContactIterations;
                const bool runFluidSolve = hasFluidWork && iteration < fluidIterations;
                const bool runBallJoints = ballJointCount > 0u && iteration < rigidJointIterations;
                const bool runHingeJoints =
                    hingeJointCount > 0u && iteration < rigidJointIterations;
                const bool runSliderJoints =
                    sliderJointCount > 0u && iteration < rigidJointIterations;
                const bool runRigidContacts =
                    useInitialRigidContactSolve && iteration < rigidContactIterations;
                const bool needContactSoftApply = runSoftContacts || runSoftRigidContacts;
                const bool needRigidApply       = runSoftRigidContacts || runRigidContacts ||
                                            runBallJoints || runHingeJoints || runSliderJoints;
                const bool needJointOnlyRigidConstants =
                    runBallJoints || runHingeJoints || runSliderJoints;

                if (runFluidSolve &&
                    !mImpl->passDispatcher.buildFluidNeighborPairs(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: BuildFluidNeighborPairs dispatch.");
                    return false;
                }
                if (runFluidSolve &&
                    !mImpl->passDispatcher.computeFluidDensityConstraints(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ComputeFluidDensityConstraints dispatch.");
                    return false;
                }
                if (runFluidSolve &&
                    !mImpl->passDispatcher.computeFluidDeltaPositions(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ComputeFluidDeltaPositions dispatch.");
                    return false;
                }
                if (runFluidSolve &&
                    !mImpl->passDispatcher.applyFluidDeltaPositions(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyFluidDeltaPositions dispatch.");
                    return false;
                }
                if (runFluidSolve &&
                    !mImpl->passDispatcher.clampFluidBoundary(computeBackend.computeContext,
                                                              mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClampFluidBoundary dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.solveSoftEdgeConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           softEdgeCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSoftEdgeConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal &&
                    !mImpl->passDispatcher.applySoftEdgeCorrections(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplySoftEdgeCorrections dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.solveSoftTetConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           softTetCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSoftTetConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal &&
                    !mImpl->passDispatcher.applySoftTetCorrections(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplySoftTetCorrections dispatch.");
                    return false;
                }
                if (runSoftContacts &&
                    !mImpl->passDispatcher.generateParticleExplicitContacts(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: GenerateParticleExplicitContacts dispatch.");
                    return false;
                }
                if (runSoftContacts &&
                    !mImpl->passDispatcher.compactParticleExplicitContacts(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: CompactSoftContacts dispatch.");
                    return false;
                }
                if (runSoftRigidContacts &&
                    !mImpl->passDispatcher.generateParticleRigidContacts(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: GenerateParticleRigidContacts dispatch.");
                    return false;
                }
                if (runSoftRigidContacts &&
                    !mImpl->passDispatcher.compactParticleRigidContacts(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: CompactSoftRigidContacts dispatch.");
                    return false;
                }
                if ((runSoftContacts || runSoftRigidContacts) &&
                    !mImpl->passDispatcher.prepareParticleActiveIndirectArgs(
                        computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: PrepareParticleActiveIndirectArgs dispatch.");
                    return false;
                }
                if (runSoftContacts &&
                    !mImpl->passDispatcher.solveParticleExplicitContacts(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveParticleExplicitContacts dispatch.");
                    return false;
                }
                if (runSoftRigidContacts &&
                    !mImpl->passDispatcher.solveParticleRigidContacts(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveParticleRigidContacts dispatch.");
                    return false;
                }
                if (needJointOnlyRigidConstants &&
                    !mImpl->passDispatcher.updateRigidDispatchConstants(
                        computeBackend.computeContext, constants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: UpdateRigidDispatchConstants dispatch.");
                    return false;
                }
                if (runBallJoints && !mImpl->passDispatcher.solveBallJointConstraints(
                                         computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveBallJointConstraints dispatch.");
                    return false;
                }
                if (runHingeJoints && !mImpl->passDispatcher.solveHingeJointConstraints(
                                          computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveHingeJointConstraints dispatch.");
                    return false;
                }
                if (runSliderJoints && !mImpl->passDispatcher.solveSliderJointConstraints(
                                           computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSliderJointConstraints dispatch.");
                    return false;
                }
                if (runRigidContacts && !mImpl->passDispatcher.finalRigidContactDepenetration(
                                            computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveRigidContactDepenetration dispatch.");
                    return false;
                }
                if (needContactSoftApply &&
                    !mImpl->passDispatcher.applyParticlePositionCorrections(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyParticlePositionCorrections dispatch.");
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

        if (rigidContactIterations > 0u && !useInitialRigidContactSolve &&
            !mImpl->passDispatcher.resetRigidContactVelocityAggregates(
                computeBackend.computeContext, mImpl->sceneState, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: InitRigidContactVelocities dispatch.");
            return false;
        }

        if (!mImpl->passDispatcher.updateRigidVelocities(
                computeBackend.computeContext, mImpl->sceneState, rigidBodyCount, constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateRigidVelocities dispatch.");
            return false;
        }

        if (!mImpl->passDispatcher.updateParticleVelocities(
                computeBackend.computeContext, mImpl->sceneState, particleCount, particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateParticleVelocities dispatch.");
            return false;
        }
        if (hasFluidWork &&
            !mImpl->passDispatcher.projectFluidBoundaryVelocities(
                computeBackend.computeContext, mImpl->sceneState, particleConstants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ProjectFluidBoundaryVelocities dispatch.");
            return false;
        }
        if (hasFluidWork &&
            !mImpl->passDispatcher.buildFluidNeighborPairs(computeBackend.computeContext,
                                                           mImpl->sceneState, particleConstants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: BuildFluidNeighborPairs post-update dispatch.");
            return false;
        }
        if (hasFluidWork &&
            !mImpl->passDispatcher.computeFluidVorticity(computeBackend.computeContext,
                                                         mImpl->sceneState, particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ComputeFluidVorticity dispatch.");
            return false;
        }
        if (hasFluidWork &&
            !mImpl->passDispatcher.applyFluidVorticityConfinement(
                computeBackend.computeContext, mImpl->sceneState, particleConstants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ApplyFluidVorticityConfinement dispatch.");
            return false;
        }
        if (hasFluidWork &&
            !mImpl->passDispatcher.buildFluidRenderAnisotropy(computeBackend.computeContext,
                                                              mImpl->sceneState, particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: BuildFluidRenderAnisotropy dispatch.");
            return false;
        }
        if (hasSoftSoftContactWork && softContactIterations > 0u &&
            !mImpl->passDispatcher.solveParticleContactVelocities(computeBackend.computeContext,
                                                                  mImpl->sceneState, particleCount,
                                                                  softContactIterations))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: SolveParticleContactVelocities dispatch.");
            return false;
        }
        if (hasSoftRigidContactWork && softContactIterations > 0u &&
            !mImpl->passDispatcher.solveParticleRigidContactVelocities(
                computeBackend.computeContext, mImpl->sceneState, particleCount, rigidBodyCount,
                softContactIterations, constants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: SolveParticleRigidContactVelocities dispatch.");
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

        const std::uint32_t rigidVelocityIterations =
            std::max(rigidContactIterations, rigidJointIterations);
        if (rigidVelocityIterations > 0u &&
            !mImpl->passDispatcher.solveRigidContactVelocities(
                computeBackend.computeContext, mImpl->sceneState, rigidBodyCount,
                rigidContactIterations, rigidJointIterations, constants))
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

    (void)runSoftPositionsCudaInteropProbe(
        mImpl->softPositionsCudaInteropProbe, computeBackend.renderDevice,
        computeBackend.computeContext, mImpl->sceneState.softPositionsInvMassSharedBuffer(),
        frameContext);

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
    if (!mImpl->sceneState.readbackPredictedParticleStateBlocking(computeBackend.computeContext,
                                                                  world, particleCount))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: readbackPredictedParticleStateBlocking.");
        return false;
    }

    return true;
}

bool PhysicsSolver::validateGpuMetaBlocking()
{
    if (!mInitialized || !hasPhysicsGpuBackend(mDevice))
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
        GpuParticleNeighborMeta particleNeighborMeta{};
        if (!mImpl->sceneState.readbackSoftNeighborMetaBlocking(computeBackend.computeContext,
                                                                particleNeighborMeta))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::validateGpuMetaBlocking failed: particle neighbor meta readback.");
            return false;
        }
        if (particleNeighborMeta.particleParticleCandidateOverflow != 0u ||
            particleNeighborMeta.particleRigidCandidateOverflow != 0u ||
            particleNeighborMeta.fluidBoundaryCandidateOverflow != 0u)
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver validation failed: particle candidate overflow "
                "(particle-particle required=",
                particleNeighborMeta.requiredParticleParticleCandidateCount,
                ", particle-rigid required=",
                particleNeighborMeta.requiredParticleRigidCandidateCount,
                ", fluid-boundary required=",
                particleNeighborMeta.requiredFluidBoundaryCandidateCount,
                ", particle-rigid capacity=", mImpl->sceneState.particleCandidatePairCapacity(),
                ", fluid-boundary capacity=",
                mImpl->sceneState.fluidBoundaryCandidatePairCapacity(), ").");
            return false;
        }
    }

    return true;
}

PhysicsGpuSceneView PhysicsSolver::gpuSceneView() const noexcept
{
    return mImpl != nullptr ? mImpl->sceneState.sceneView() : PhysicsGpuSceneView{};
}

const gpu::SharedExportBuffer *PhysicsSolver::softPositionsInvMassSharedBuffer() const noexcept
{
    return mImpl != nullptr ? &mImpl->sceneState.softPositionsInvMassSharedBuffer() : nullptr;
}

} // namespace cressim::neo::physics
