#include "physics/physics_solver.h"

#include "common/logger.h"
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

std::uint32_t buildUniqueQueueFamilyIndices(const Diligent::IDeviceContext *firstContext,
                                            const Diligent::IDeviceContext *secondContext,
                                            std::array<std::uint32_t, 2> &outQueueFamilyIndices)
{
    std::uint32_t count      = 0u;
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

} // namespace

struct PhysicsSolver::Impl
{
    Impl(gpu::GpuDevice &deviceIn, const PhysicsSolverDesc &descIn)
        : mDevice(deviceIn), mDesc(descIn)
    {
    }

    gpu::GpuDevice &mDevice;
    PhysicsSolverDesc mDesc{};
    PhysicsSceneGpuState sceneState;
    PhysicsPassDispatcher passDispatcher;
    bool lastStepHadRigidBroadPhaseWork             = false;
    bool lastStepHadSoftPairWork                    = false;
    std::uint64_t lastAppliedRigidBindingGeneration = 0u;
    std::uint64_t lastAppliedSoftBindingGeneration  = 0u;
    bool mInitialized                               = false;
};

GpuPhysicsSolverConfig makeSolverConfig(const PhysicsSolverDesc &desc)
{
    GpuPhysicsSolverConfig config{};
    config.contact0 = {desc.contact.slop, desc.contact.manifoldMergeSlopMultiplier,
                       desc.contact.softRelaxation, desc.contact.softMaxCorrectionPerIteration};
    config.contact1 = {desc.contact.restitutionVelocityThreshold,
                       desc.contact.restitutionPenetrationSlopMultiplier,
                       desc.contact.rigidRestitutionVelocityThreshold,
                       desc.contact.positionFrictionMinDistance};
    config.rigid0   = {desc.rigid.depenetrationBaumgarte, desc.rigid.depenetrationRelaxation,
                       desc.rigid.depenetrationMaxCorrectionPerIteration,
                       desc.rigid.velocityPenetrationStiffness};
    config.rigid1   = {desc.rigid.maxTranslationCorrectionPerIteration,
                       desc.rigid.maxRotationCorrectionPerIteration,
                       desc.rigid.maxLinearVelocityCorrectionPerIteration,
                       desc.rigid.maxAngularVelocityCorrectionPerIteration};
    config.joints0  = {desc.joints.ballRelaxation, desc.joints.articulatedRelaxation,
                       desc.joints.maxError, desc.joints.maxTranslationCorrection};
    config.joints1  = {desc.joints.maxAngularCorrection, desc.joints.regularization,
                       desc.joints.angularRegularization, desc.joints.minXpbdDt};
    config.soft0    = {desc.soft.internalRelaxation, desc.soft.reserved0, desc.soft.reserved1,
                       desc.soft.reserved2};
    config.soft1    = {desc.soft.reserved3, desc.soft.reserved4, desc.soft.reserved5,
                       desc.soft.reserved6};
    config.fluid0   = {desc.fluid.constraintRelaxation, desc.fluid.positionRelaxation,
                       desc.fluid.boundaryDensityScale, desc.fluid.boundaryDeltaScale};
    config.fluid1   = {desc.fluid.reserved0, desc.fluid.reserved1, desc.fluid.reserved2,
                       desc.fluid.reserved3};
    return config;
}

PhysicsSolver::PhysicsSolver(gpu::GpuDevice &device, const PhysicsSolverDesc &desc)
    : mImpl(std::make_unique<Impl>(device, desc))
{
}

PhysicsSolver::~PhysicsSolver() = default;

bool PhysicsSolver::initialize()
{
    shutdown();

    if (!hasPhysicsGpuBackend(mImpl->mDevice))
    {
        mImpl->mInitialized = true;
        return true;
    }

    gpu::GpuComputeBackendContext computeContext{};
    if (!mImpl->mDevice.tryGetPhysicsBackendContext(computeContext) ||
        computeContext.renderDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("PhysicsSolver: failed to get physics GPU context.");
        return false;
    }

    mImpl = std::make_unique<Impl>(mImpl->mDevice, mImpl->mDesc);
    if (!mImpl->passDispatcher.initialize(mImpl->mDevice, computeContext.contextId))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver: failed to initialize physics pass dispatcher.");
        return false;
    }

    mImpl->mInitialized = true;
    return true;
}

void PhysicsSolver::shutdown()
{
    mImpl               = std::make_unique<Impl>(mImpl->mDevice, mImpl->mDesc);
    mImpl->mInitialized = false;
}

bool PhysicsSolver::syncWorldState(PhysicsWorld &world)
{
    if (!mImpl->mInitialized)
    {
        return false;
    }

    if (!hasPhysicsGpuBackend(mImpl->mDevice))
    {
        return true;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    gpu::GpuGraphicsBackendContext graphicsBackend{};
    if (!mImpl->mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        !mImpl->mDevice.tryGetGraphicsBackendContext(graphicsBackend) ||
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
    const std::vector<DeformableDistanceConstraint> &distanceConstraints =
        world.distanceConstraints();
    const std::vector<DeformableBendConstraint> &bendConstraints     = world.bendConstraints();
    const std::vector<DeformableVolumeConstraint> &volumeConstraints = world.volumeConstraints();
    const std::vector<StrandSegmentConstraint> &strandSegments       = world.strandSegments();
    const std::vector<StrandJointConstraint> &strandJoints           = world.strandJoints();
    const std::vector<StrandDistanceConstraint> &strandDistanceConstraints =
        world.strandDistanceConstraints();
    const std::vector<RigidParticleAttachmentConstraint> &rigidParticleAttachments =
        world.rigidParticleAttachments();
    const std::vector<StrandRigidAttachmentConstraint> &strandRigidAttachments =
        world.strandRigidAttachments();
    const std::vector<RigidDistanceConstraint> &rigidDistanceConstraints =
        world.rigidDistanceConstraints();
    const std::vector<RoutedCableConstraint> &routedCableConstraints =
        world.routedCableConstraints();
    const std::vector<RoutedCableRoutePoint> &routedCableRoutePoints =
        world.routedCableRoutePoints();
    const SoftRenderDataHost &softRenderData   = world.softRenderData();
    const CurveRenderDataHost &curveRenderData = world.curveRenderData();
    const RigidJointSceneHost &rigidJoints     = world.rigidJointScene();
    const std::uint32_t fluidCount             = world.fluidCount();
    const std::uint32_t particleCount          = static_cast<std::uint32_t>(particles.size());
    const std::uint32_t softEdgeCount      = static_cast<std::uint32_t>(distanceConstraints.size());
    const std::uint32_t softBendCount      = static_cast<std::uint32_t>(bendConstraints.size());
    const std::uint32_t softTetCount       = static_cast<std::uint32_t>(volumeConstraints.size());
    const std::uint32_t strandSegmentCount = static_cast<std::uint32_t>(strandSegments.size());
    const std::uint32_t strandJointCount   = static_cast<std::uint32_t>(strandJoints.size());
    const std::uint32_t strandDistanceCount =
        static_cast<std::uint32_t>(strandDistanceConstraints.size());
    const std::uint32_t routedCableCount =
        static_cast<std::uint32_t>(routedCableConstraints.size());
    const std::uint32_t rigidParticleAttachmentCount =
        static_cast<std::uint32_t>(rigidParticleAttachments.size());
    const std::uint32_t strandRigidAttachmentCount =
        static_cast<std::uint32_t>(strandRigidAttachments.size());
    const std::uint32_t rigidDistanceConstraintCount =
        static_cast<std::uint32_t>(rigidDistanceConstraints.size());
    std::uint32_t routedCableDebugSegmentCount = 0u;
    for (const RoutedCableConstraint &constraint : routedCableConstraints)
    {
        if (constraint.routePointCount > 1u)
        {
            routedCableDebugSegmentCount += constraint.routePointCount - 1u;
        }
    }
    const std::uint32_t ballJointCount = static_cast<std::uint32_t>(rigidJoints.ball.size());
    const std::uint32_t sphericalJointCount =
        static_cast<std::uint32_t>(rigidJoints.spherical.size());
    const std::uint32_t hingeJointCount  = static_cast<std::uint32_t>(rigidJoints.hinge.size());
    const std::uint32_t sliderJointCount = static_cast<std::uint32_t>(rigidJoints.slider.size());
    const std::uint32_t softRenderTriangleCount =
        static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size());
    const std::uint32_t curveRenderCount =
        static_cast<std::uint32_t>(curveRenderData.descriptors.size());
    const std::uint32_t softBodyBoundsChunkCount = world.softBodyBoundsChunkCount();
    const std::uint32_t suturingParticleCount    = world.suturingParticleCount();
    const float particleGridCellSize             = world.particleGridCellSize();
    std::array<std::uint32_t, 2> sharedQueueFamilyIndices{};
    const std::uint32_t sharedQueueFamilyIndexCount = buildUniqueQueueFamilyIndices(
        computeBackend.computeContext, graphicsBackend.graphicsContext, sharedQueueFamilyIndices);
    const bool hasSoftData = particleCount > 0u || softEdgeCount > 0u || softBendCount > 0u ||
                             softTetCount > 0u || strandSegmentCount > 0u ||
                             strandJointCount > 0u || strandDistanceCount > 0u;
    if (rigidBodyCount == 0u && !hasSoftData)
    {
        return true;
    }

    if (!mImpl->sceneState.ensureCapacity(
            computeBackend.renderDevice, rigidBodyCount, colliderCount, particleCount, fluidCount,
            static_cast<std::uint32_t>(world.particleContactMaterials().size()),
            static_cast<std::uint32_t>(fluidMaterials.size()), softEdgeCount, softBendCount,
            softTetCount, strandSegmentCount, strandJointCount, strandDistanceCount, ballJointCount,
            sphericalJointCount, hingeJointCount, sliderJointCount, rigidParticleAttachmentCount,
            strandRigidAttachmentCount, rigidDistanceConstraintCount,
            static_cast<std::uint32_t>(softRenderData.fallbackNormals.size()),
            static_cast<std::uint32_t>(softRenderData.vertexTriangleIndices.size()),
            softRenderTriangleCount,
            static_cast<std::uint32_t>(softRenderData.softBodyParticleRanges.size()),
            softBodyBoundsChunkCount, static_cast<std::uint32_t>(world.suturingPairs().size()),
            world.reservedSuturingPathHeaderCount(), world.reservedSuturingPathNodeCount(),
            routedCableCount, static_cast<std::uint32_t>(routedCableRoutePoints.size()),
            routedCableDebugSegmentCount,
            static_cast<std::uint32_t>(curveRenderData.descriptors.size()),
            static_cast<std::uint32_t>(curveRenderData.particleIndices.size()),
            [&curveRenderData]()
            {
                std::uint32_t totalVertexCount = 0u;
                for (const CurveRenderDescriptorHost &descriptor : curveRenderData.descriptors)
                {
                    totalVertexCount =
                        std::max(totalVertexCount, descriptor.vertexBase + descriptor.vertexCount);
                }
                return totalVertexCount;
            }(),
            gpu::contextMaskForId(computeBackend.contextId) |
                gpu::contextMaskForId(graphicsBackend.contextId),
            sharedQueueFamilyIndices.data(), sharedQueueFamilyIndexCount,
            mImpl->mDevice.supportsNativePhysicsFloatAtomics()))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::syncWorldState failed: ensureCapacity.");
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
        CRESSIM_LOG_ERROR("PhysicsSolver::syncWorldState failed: uploadWorldState.");
        return false;
    }

    return true;
}

bool PhysicsSolver::step(const common::FrameContext &frameContext, PhysicsWorld &world)
{
    if (!syncWorldState(world))
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mImpl->mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        CRESSIM_LOG_ERROR("PhysicsSolver::step failed: missing physics backend context.");
        return false;
    }

    const std::uint32_t rigidBodyCount = world.rigidBodyCount();
    const std::uint32_t colliderCount  = world.colliderCount();
    const ParticleSoAHost &particles   = world.particles();
    const std::vector<DeformableDistanceConstraint> &distanceConstraints =
        world.distanceConstraints();
    const std::vector<DeformableBendConstraint> &bendConstraints     = world.bendConstraints();
    const std::vector<DeformableVolumeConstraint> &volumeConstraints = world.volumeConstraints();
    const std::vector<StrandSegmentConstraint> &strandSegments       = world.strandSegments();
    const std::vector<StrandJointConstraint> &strandJoints           = world.strandJoints();
    const std::vector<StrandDistanceConstraint> &strandDistanceConstraints =
        world.strandDistanceConstraints();
    const std::vector<RigidParticleAttachmentConstraint> &rigidParticleAttachments =
        world.rigidParticleAttachments();
    const std::vector<StrandRigidAttachmentConstraint> &strandRigidAttachments =
        world.strandRigidAttachments();
    const std::vector<RigidDistanceConstraint> &rigidDistanceConstraints =
        world.rigidDistanceConstraints();
    const std::vector<RoutedCableConstraint> &routedCableConstraints =
        world.routedCableConstraints();
    const RigidJointSceneHost &rigidJoints     = world.rigidJointScene();
    const SoftRenderDataHost &softRenderData   = world.softRenderData();
    const CurveRenderDataHost &curveRenderData = world.curveRenderData();
    const std::uint32_t fluidCount             = world.fluidCount();
    const std::uint32_t particleCount          = static_cast<std::uint32_t>(particles.size());
    const std::uint32_t softEdgeCount      = static_cast<std::uint32_t>(distanceConstraints.size());
    const std::uint32_t softBendCount      = static_cast<std::uint32_t>(bendConstraints.size());
    const std::uint32_t softTetCount       = static_cast<std::uint32_t>(volumeConstraints.size());
    const std::uint32_t strandSegmentCount = static_cast<std::uint32_t>(strandSegments.size());
    const std::uint32_t strandJointCount   = static_cast<std::uint32_t>(strandJoints.size());
    const std::uint32_t strandDistanceCount =
        static_cast<std::uint32_t>(strandDistanceConstraints.size());
    const std::uint32_t routedCableCount =
        static_cast<std::uint32_t>(routedCableConstraints.size());
    const std::uint32_t ballJointCount = static_cast<std::uint32_t>(rigidJoints.ball.size());
    const std::uint32_t sphericalJointCount =
        static_cast<std::uint32_t>(rigidJoints.spherical.size());
    const std::uint32_t hingeJointCount  = static_cast<std::uint32_t>(rigidJoints.hinge.size());
    const std::uint32_t sliderJointCount = static_cast<std::uint32_t>(rigidJoints.slider.size());
    const std::uint32_t rigidParticleAttachmentCount =
        static_cast<std::uint32_t>(rigidParticleAttachments.size());
    const std::uint32_t strandRigidAttachmentCount =
        static_cast<std::uint32_t>(strandRigidAttachments.size());
    const std::uint32_t rigidDistanceConstraintCount =
        static_cast<std::uint32_t>(rigidDistanceConstraints.size());
    const std::uint32_t softRenderTriangleCount =
        static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size());
    const std::uint32_t curveRenderCount =
        static_cast<std::uint32_t>(curveRenderData.descriptors.size());
    const std::uint32_t softBodyBoundsChunkCount = world.softBodyBoundsChunkCount();
    const std::uint32_t suturingParticleCount    = world.suturingParticleCount();
    const float particleGridCellSize             = world.particleGridCellSize();

    const std::uint32_t substeps = std::max<std::uint32_t>(mImpl->mDesc.substeps, 1u);
    const std::uint32_t defaultIterations =
        std::max<std::uint32_t>(mImpl->mDesc.defaultIterations, 1u);
    const std::uint32_t fluidIterations =
        resolveIterations(mImpl->mDesc.fluidIterations, defaultIterations);
    const std::uint32_t softInternalIterations =
        resolveIterations(mImpl->mDesc.softInternalIterations, defaultIterations);
    const std::uint32_t softContactIterations =
        resolveIterations(mImpl->mDesc.softContactIterations, defaultIterations);
    const std::uint32_t rigidJointIterations =
        resolveIterations(mImpl->mDesc.rigidJointIterations, defaultIterations);
    const std::uint32_t rigidContactIterations =
        resolveIterations(mImpl->mDesc.rigidRigidContactIterations, defaultIterations);
    const std::uint32_t maxPositionPhaseIterations =
        std::max(std::max(std::max(fluidIterations, softInternalIterations), softContactIterations),
                 std::max(rigidJointIterations, rigidContactIterations));
    const float substepDt = frameContext.deltaSeconds / static_cast<float>(substeps);

    if (!mImpl->passDispatcher.updateSolverConfig(computeBackend.computeContext,
                                                  makeSolverConfig(mImpl->mDesc)))
    {
        CRESSIM_LOG_ERROR("PhysicsSolver: failed to update solver configuration.");
        return false;
    }

    for (std::uint32_t substep = 0; substep < substeps; ++substep)
    {
        GpuRigidDispatchConstants constants{};
        constants.gravity = {mImpl->mDesc.gravity.x, mImpl->mDesc.gravity.y, mImpl->mDesc.gravity.z,
                             0.0f};
        constants.dt      = substepDt;
        constants.rigidBodyCount        = rigidBodyCount;
        constants.colliderCount         = colliderCount;
        constants.candidatePairCapacity = mImpl->sceneState.candidatePairCapacity();
        constants.contactCapacity       = mImpl->sceneState.rigidContactCapacity();
        constants.reserved0             = routedCableCount;
        GpuParticleDispatchConstants particleConstants{};
        particleConstants.dt                   = substepDt;
        particleConstants.gravity              = {mImpl->mDesc.gravity.x, mImpl->mDesc.gravity.y,
                                                  mImpl->mDesc.gravity.z, 0.0f};
        particleConstants.particleCount        = particleCount;
        particleConstants.rigidColliderCount   = colliderCount;
        particleConstants.particleGridCellSize = particleGridCellSize;
        particleConstants.particleCandidatePairCapacity =
            mImpl->sceneState.particleCandidatePairCapacity();
        particleConstants.fluidBoundaryCandidatePairCapacity =
            mImpl->sceneState.fluidBoundaryCandidatePairCapacity();
        particleConstants.maxFluidNeighborhood = mImpl->sceneState.maxFluidNeighborhood();
        particleConstants.particleCellRangeCapacity =
            nextPowerOfTwo(std::max<std::uint32_t>(particleCount * 2u, 1u));
        particleConstants.softEdgeCount       = softEdgeCount;
        particleConstants.softBendCount       = softBendCount;
        particleConstants.softTetCount        = softTetCount;
        particleConstants.strandSegmentCount  = strandSegmentCount;
        particleConstants.strandJointCount    = strandJointCount;
        particleConstants.strandDistanceCount = strandDistanceCount;
        particleConstants.fluidIterations     = fluidIterations;
        particleConstants.suturingPairCount =
            static_cast<std::uint32_t>(world.suturingPairs().size());
        particleConstants.suturingPathHeaderCount = world.reservedSuturingPathHeaderCount();
        particleConstants.suturingPathNodeCount   = world.reservedSuturingPathNodeCount();
        particleConstants.suturingParticleCount   = suturingParticleCount;
        particleConstants.maxSuturingCandidatesPerParticle = kMaxSuturingCandidatesPerParticle;
        particleConstants.maxSuturingNodesPerPath          = world.maxSuturingNodesPerPath();

        const bool hasParticleNeighborWork = particleCount > 0u;
        const bool hasFluidWork            = fluidCount > 0u && particleCount > 0u;
        const bool hasFluidBoundaryWork    = hasFluidWork && colliderCount > 0u;
        const bool hasSoftInternalWork =
            particleCount > 0u &&
            (softEdgeCount > 0u || softBendCount > 0u || softTetCount > 0u ||
             strandSegmentCount > 0u || strandJointCount > 0u || strandDistanceCount > 0u);
        const bool hasSoftContactSolveWork       = softContactIterations > 0u;
        const bool hasSoftSoftContactWork        = hasSoftContactSolveWork && particleCount > 1u;
        const bool hasParticleRigidCandidateWork = particleCount > 0u && colliderCount > 0u;
        const bool hasSoftRigidContactWork =
            hasSoftContactSolveWork && hasParticleRigidCandidateWork;
        const bool hasSuturingCouplingWork =
            suturingParticleCount > 0u && particleConstants.suturingPairCount > 0u;
        const bool hasRoutedCableWork             = routedCableCount > 0u;
        const bool hasRigidParticleAttachmentWork = rigidParticleAttachmentCount > 0u;
        const bool hasStrandRigidAttachmentWork   = strandRigidAttachmentCount > 0u;
        const bool hasRigidDistanceConstraintWork = rigidDistanceConstraintCount > 0u;
        const bool hasParticleBroadPhaseWork =
            hasSoftSoftContactWork || hasFluidWork || hasParticleRigidCandidateWork;
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
            rigidContactIterations > 0u &&
            (hasRigidBroadPhaseWork || hasParticleRigidCandidateWork);
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
            }
        }

        if (!mImpl->passDispatcher.predictSoft(computeBackend.computeContext, mImpl->sceneState,
                                               particleCount, particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SoftPredict dispatch.");
            return false;
        }

        if (particleCount > 0u && rigidBodyCount > 0u &&
            !mImpl->passDispatcher.syncRigidProxyParticles(
                computeBackend.computeContext, mImpl->sceneState, particleCount, particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SyncRigidProxyParticles dispatch.");
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
            if (hasParticleRigidCandidateWork &&
                !mImpl->passDispatcher.buildParticleRigidCandidatePairs(
                    computeBackend.computeContext, mImpl->sceneState, particleCount,
                    particleConstants))
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
            if ((hasSoftSoftContactWork || hasParticleRigidCandidateWork) &&
                !mImpl->passDispatcher.prepareParticleCandidateIndirectArgs(
                    computeBackend.computeContext, mImpl->sceneState))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: PrepareParticleCandidateIndirectArgs dispatch.");
                return false;
            }
        }

        if (suturingParticleCount > 0u && particleConstants.suturingPairCount > 0u)
        {
            if (!mImpl->passDispatcher.clearSuturingCandidates(
                    computeBackend.computeContext, mImpl->sceneState, suturingParticleCount,
                    particleConstants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: ClearSuturingCandidates dispatch.");
                return false;
            }
            if (hasSoftSoftContactWork &&
                !mImpl->passDispatcher.gatherSuturingCandidates(
                    computeBackend.computeContext, mImpl->sceneState, particleConstants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: GatherSuturingCandidates dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.classifySuturingParticles(
                    computeBackend.computeContext, mImpl->sceneState, suturingParticleCount,
                    particleConstants))
            {
                CRESSIM_LOG_ERROR(
                    "PhysicsSolver::step failed: ClassifySuturingParticles dispatch.");
                return false;
            }
            if (!mImpl->passDispatcher.updateSuturingTipPaths(
                    computeBackend.computeContext, mImpl->sceneState,
                    particleConstants.suturingPairCount, particleConstants))
            {
                CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateSuturingTipPaths dispatch.");
                return false;
            }
        }

        if (useInitialRigidContactSolve)
        {
            const GpuProxyRigidContactMeta zeroProxyMeta{};
            computeBackend.computeContext->UpdateBuffer(
                mImpl->sceneState.transientBuffers().proxyRigidContactMetaBuffer, 0u,
                sizeof(GpuProxyRigidContactMeta), &zeroProxyMeta,
                Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
        if (useInitialRigidContactSolve && hasParticleRigidCandidateWork &&
            !mImpl->passDispatcher.generateProxyRigidContacts(computeBackend.computeContext,
                                                              mImpl->sceneState, particleConstants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: GenerateProxyRigidContacts dispatch.");
            return false;
        }
        if (useInitialRigidContactSolve && !mImpl->passDispatcher.prepareRigidIndirectArgs(
                                               computeBackend.computeContext, mImpl->sceneState))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: PrepareRigidIndirectArgs final dispatch.");
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

        const std::uint32_t softConstraintThreadCount = std::max(
            std::max(std::max(particleCount,
                              std::max(std::max(softEdgeCount, softBendCount), softTetCount)),
                     std::max(std::max(strandSegmentCount, strandJointCount), strandDistanceCount)),
            rigidParticleAttachmentCount);
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
        if (hasRoutedCableWork &&
            !mImpl->passDispatcher.clearRoutedCableConstraintState(
                computeBackend.computeContext, mImpl->sceneState, routedCableCount, constants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ClearRoutedCableConstraintState dispatch.");
            return false;
        }
        if (hasRigidParticleAttachmentWork &&
            !mImpl->passDispatcher.clearRigidParticleAttachmentConstraintState(
                computeBackend.computeContext, mImpl->sceneState, rigidParticleAttachmentCount,
                constants))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: "
                              "ClearRigidParticleAttachmentConstraintState dispatch.");
            return false;
        }
        if (hasStrandRigidAttachmentWork &&
            !mImpl->passDispatcher.clearStrandRigidAttachmentConstraintState(
                computeBackend.computeContext, mImpl->sceneState, strandRigidAttachmentCount,
                constants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ClearStrandRigidAttachmentConstraintState dispatch.");
            return false;
        }
        if (hasRigidDistanceConstraintWork &&
            !mImpl->passDispatcher.clearRigidDistanceConstraintState(
                computeBackend.computeContext, mImpl->sceneState, rigidDistanceConstraintCount,
                constants))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ClearRigidDistanceConstraintState dispatch.");
            return false;
        }
        if (hingeJointCount > 0u &&
            !mImpl->passDispatcher.updateHingeJointRuntimeState(computeBackend.computeContext,
                                                                mImpl->sceneState, hingeJointCount))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateHingeJointRuntimeState dispatch.");
            return false;
        }
        if (sliderJointCount > 0u &&
            !mImpl->passDispatcher.updateSliderJointRuntimeState(
                computeBackend.computeContext, mImpl->sceneState, sliderJointCount))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: UpdateSliderJointRuntimeState dispatch.");
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
        if (sphericalJointCount > 0u &&
            !mImpl->passDispatcher.clearSphericalJointConstraintState(
                computeBackend.computeContext, mImpl->sceneState, sphericalJointCount))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: ClearSphericalJointConstraintState dispatch.");
            return false;
        }

        const bool hasAnyPositionSolveWork =
            hasFluidWork || (hasSoftInternalWork && softInternalIterations > 0u) ||
            (hasSoftSoftContactWork && softContactIterations > 0u) ||
            (hasSoftRigidContactWork && softContactIterations > 0u) || hasSuturingCouplingWork ||
            hasRoutedCableWork || hasRigidParticleAttachmentWork || hasStrandRigidAttachmentWork ||
            hasRigidDistanceConstraintWork || useInitialRigidContactSolve ||
            ((ballJointCount > 0u || sphericalJointCount > 0u || hingeJointCount > 0u ||
              sliderJointCount > 0u) &&
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
                const bool runSphericalJoints =
                    sphericalJointCount > 0u && iteration < rigidJointIterations;
                const bool runHingeJoints =
                    hingeJointCount > 0u && iteration < rigidJointIterations;
                const bool runSliderJoints =
                    sliderJointCount > 0u && iteration < rigidJointIterations;
                const bool runRigidContacts =
                    useInitialRigidContactSolve && iteration < rigidContactIterations;
                const bool needContactSoftApply = runSoftContacts || runSoftRigidContacts;
                const bool needAttachmentApply =
                    hasRigidParticleAttachmentWork || hasStrandRigidAttachmentWork;
                const bool needRoutedCableApply   = hasRoutedCableWork;
                const bool needRigidDistanceApply = hasRigidDistanceConstraintWork;
                const bool needRigidApply = runSoftRigidContacts || runRigidContacts ||
                                            runBallJoints || runSphericalJoints || runHingeJoints ||
                                            runSliderJoints || needAttachmentApply ||
                                            needRoutedCableApply || needRigidDistanceApply;
                const bool needJointOnlyRigidConstants =
                    runBallJoints || runSphericalJoints || runHingeJoints || runSliderJoints;

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
                if (runSoftInternal && !mImpl->passDispatcher.solveStrandSegmentConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           strandSegmentCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveStrandSegmentConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal &&
                    !mImpl->passDispatcher.applyStrandSegmentCorrections(
                        computeBackend.computeContext, mImpl->sceneState,
                        std::max(particleCount, strandSegmentCount), particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyStrandSegmentCorrections dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.solveSoftBendConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           softBendCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSoftBendConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal &&
                    !mImpl->passDispatcher.applySoftBendCorrections(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplySoftBendCorrections dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.solveStrandJointConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           strandJointCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveStrandJointConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.applyStrandJointCorrections(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           strandSegmentCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyStrandJointCorrections dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.solveStrandDistanceConstraints(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           strandDistanceCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveStrandDistanceConstraints dispatch.");
                    return false;
                }
                if (runSoftInternal && !mImpl->passDispatcher.applyStrandDistanceCorrections(
                                           computeBackend.computeContext, mImpl->sceneState,
                                           particleCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyStrandDistanceCorrections dispatch.");
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
                if (runSphericalJoints && !mImpl->passDispatcher.solveSphericalJointConstraints(
                                              computeBackend.computeContext, mImpl->sceneState))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSphericalJointConstraints dispatch.");
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
                if (hasRigidParticleAttachmentWork &&
                    !mImpl->passDispatcher.solveRigidParticleAttachmentConstraints(
                        computeBackend.computeContext, mImpl->sceneState,
                        rigidParticleAttachmentCount, constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: "
                                      "SolveRigidParticleAttachmentConstraints dispatch.");
                    return false;
                }
                if (hasStrandRigidAttachmentWork &&
                    !mImpl->passDispatcher.solveStrandRigidAttachmentConstraints(
                        computeBackend.computeContext, mImpl->sceneState,
                        strandRigidAttachmentCount, constants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: "
                                      "SolveStrandRigidAttachmentConstraints dispatch.");
                    return false;
                }
                if (hasRoutedCableWork && !mImpl->passDispatcher.solveRoutedCableConstraints(
                                              computeBackend.computeContext, mImpl->sceneState,
                                              routedCableCount, constants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveRoutedCableConstraints dispatch.");
                    return false;
                }
                if (hasRigidDistanceConstraintWork &&
                    !mImpl->passDispatcher.solveRigidDistanceConstraints(
                        computeBackend.computeContext, mImpl->sceneState,
                        rigidDistanceConstraintCount, constants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveRigidDistanceConstraints dispatch.");
                    return false;
                }
                if ((needContactSoftApply || needAttachmentApply) &&
                    !mImpl->passDispatcher.applyParticlePositionCorrections(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyParticlePositionCorrections dispatch.");
                    return false;
                }
                if (hasStrandRigidAttachmentWork &&
                    !mImpl->passDispatcher.applyStrandRigidAttachmentCorrections(
                        computeBackend.computeContext, mImpl->sceneState, strandSegmentCount,
                        particleConstants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: "
                                      "ApplyStrandRigidAttachmentCorrections dispatch.");
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
                if (needRigidApply && particleCount > 0u && rigidBodyCount > 0u &&
                    !mImpl->passDispatcher.syncRigidProxyParticles(computeBackend.computeContext,
                                                                   mImpl->sceneState, particleCount,
                                                                   particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SyncRigidProxyParticles iterative dispatch.");
                    return false;
                }
                if (hasSuturingCouplingWork && !mImpl->passDispatcher.assignSuturingInsideParticles(
                                                   computeBackend.computeContext, mImpl->sceneState,
                                                   suturingParticleCount, particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: AssignSuturingInsideParticles dispatch.");
                    return false;
                }
                if (hasSuturingCouplingWork &&
                    !mImpl->passDispatcher.solveSuturingNodePathConstraints(
                        computeBackend.computeContext, mImpl->sceneState, suturingParticleCount,
                        particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SolveSuturingNodePathConstraints dispatch.");
                    return false;
                }
                if (hasSuturingCouplingWork &&
                    !mImpl->passDispatcher.applyParticlePositionCorrections(
                        computeBackend.computeContext, mImpl->sceneState, particleConstants))
                {
                    CRESSIM_LOG_ERROR("PhysicsSolver::step failed: "
                                      "ApplyParticlePositionCorrections suturing dispatch.");
                    return false;
                }
                if (hasSuturingCouplingWork && rigidBodyCount > 0u &&
                    !mImpl->passDispatcher.applyRigidCorrections(computeBackend.computeContext,
                                                                 mImpl->sceneState, rigidBodyCount,
                                                                 constants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: ApplyRigidCorrections suturing dispatch.");
                    return false;
                }
                if (hasSuturingCouplingWork && rigidBodyCount > 0u && particleCount > 0u &&
                    !mImpl->passDispatcher.syncRigidProxyParticles(computeBackend.computeContext,
                                                                   mImpl->sceneState, particleCount,
                                                                   particleConstants))
                {
                    CRESSIM_LOG_ERROR(
                        "PhysicsSolver::step failed: SyncRigidProxyParticles suturing dispatch.");
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
            !mImpl->passDispatcher.solveParticleContactVelocities(
                computeBackend.computeContext, mImpl->sceneState, particleCount, rigidBodyCount,
                softContactIterations, constants))
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
        if (!mImpl->passDispatcher.skinSoftRenderVertices(
                computeBackend.computeContext, mImpl->sceneState,
                static_cast<std::uint32_t>(softRenderData.vertexBindings.size())))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: SkinSoftRenderVertices dispatch.");
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
        if (!mImpl->passDispatcher.updateCurveRenderData(computeBackend.computeContext,
                                                         mImpl->sceneState, curveRenderCount))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::step failed: UpdateCurveRenderData dispatch.");
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

        if (hingeJointCount > 0u &&
            !mImpl->passDispatcher.updateHingeJointRuntimeState(computeBackend.computeContext,
                                                                mImpl->sceneState, hingeJointCount))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: final UpdateHingeJointRuntimeState dispatch.");
            return false;
        }
        if (sliderJointCount > 0u &&
            !mImpl->passDispatcher.updateSliderJointRuntimeState(
                computeBackend.computeContext, mImpl->sceneState, sliderJointCount))
        {
            CRESSIM_LOG_ERROR(
                "PhysicsSolver::step failed: final UpdateSliderJointRuntimeState dispatch.");
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

    if (!mImpl->mDesc.enableBlockingReadback)
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
    if (!mImpl->mInitialized || !hasPhysicsGpuBackend(mImpl->mDevice))
    {
        return false;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mImpl->mDevice.tryGetPhysicsBackendContext(computeBackend) ||
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

        GpuProxyRigidContactMeta proxyMeta{};
        if (!mImpl->sceneState.readbackProxyRigidContactMetaBlocking(computeBackend.computeContext,
                                                                     proxyMeta))
        {
            CRESSIM_LOG_ERROR("PhysicsSolver::validateGpuMetaBlocking failed: proxy rigid contact "
                              "meta readback.");
            return false;
        }
        if (proxyMeta.overflow != 0u)
        {
            const std::uint32_t primitiveContactCount =
                broadPhaseMeta.candidatePairCount * kRigidContactsPerPair;
            const std::uint32_t proxyCapacity =
                mImpl->sceneState.rigidContactCapacity() > primitiveContactCount
                    ? (mImpl->sceneState.rigidContactCapacity() - primitiveContactCount)
                    : 0u;
            CRESSIM_LOG_ERROR(
                "PhysicsSolver validation failed: proxy rigid contact overflow "
                "(required=",
                proxyMeta.requiredContactCount, ", proxy capacity=", proxyCapacity,
                ", total rigid contact capacity=", mImpl->sceneState.rigidContactCapacity(), ").");
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

void PhysicsSolver::setGravity(const Diligent::float3 &gravity) noexcept
{
    mImpl->mDesc.gravity = gravity;
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
