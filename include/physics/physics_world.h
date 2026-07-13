#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H

#include "common/flags.h"
#include "physics/export.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace cressim::neo::physics
{

enum class PhysicsRebuildFlags : std::uint32_t
{
    None                             = 0u,
    SoftParticleLayout               = 1u << 0u,
    SoftConstraintData               = 1u << 1u,
    SuturingData                     = 1u << 2u,
    ResolvedRigidParticleAttachments = 1u << 3u,
    ResolvedStrandRigidAttachments   = 1u << 4u,
    ResolvedRigidDistanceConstraints = 1u << 5u,
    ResolvedRoutedCables             = 1u << 6u,
};

CRESSIM_NEO_DEFINE_ENUM_FLAGS(PhysicsRebuildFlags)

class CRESSIM_NEO_PHYSICS_API PhysicsWorld
{
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld &other);
    PhysicsWorld &operator=(const PhysicsWorld &other);
    PhysicsWorld(PhysicsWorld &&other) noexcept;
    PhysicsWorld &operator=(PhysicsWorld &&other) noexcept;

    void clear();

    RigidBodyState &upsertRigidBody(const RigidBodyState &state);
    bool removeRigidBody(common::EntityId entityId);
    void upsertCollider(const ColliderState &collider);
    bool removeCollider(ColliderId colliderId);
    void replaceColliders(common::EntityId entityId, const std::vector<ColliderState> &colliders);
    bool upsertSoftBody(const SoftBodyState &state);
    bool upsertStrand(const StrandState &state);
    bool upsertFluid(const FluidState &state);
    AuthoredParticleSequenceState &upsertParticleSequence(
        const AuthoredParticleSequenceState &state);
    AuthoredParticleDistanceConstraintState &upsertParticleDistanceConstraint(
        const AuthoredParticleDistanceConstraintState &state);
    bool upsertRigidParticleAttachmentConstraint(
        const AuthoredRigidParticleAttachmentConstraintState &state,
        AuthoredRigidParticleAttachmentConstraintState *outAuthored = nullptr);
    bool upsertStrandRigidAttachmentConstraint(
        const AuthoredStrandRigidAttachmentConstraintState &state,
        AuthoredStrandRigidAttachmentConstraintState *outAuthored = nullptr);
    bool upsertRigidDistanceConstraint(const AuthoredRigidDistanceConstraintState &state,
                                       AuthoredRigidDistanceConstraintState *outAuthored = nullptr);
    bool upsertRoutedCableConstraint(const AuthoredRoutedCableConstraintState &state,
                                     AuthoredRoutedCableConstraintState *outAuthored = nullptr);
    AuthoredParticleCollisionFilterState &upsertParticleCollisionFilter(
        const AuthoredParticleCollisionFilterState &state);
    AuthoredSuturingSequenceState &upsertSuturingSequence(
        const AuthoredSuturingSequenceState &state);
    bool removeSoftBody(common::EntityId entityId);
    bool removeStrand(common::EntityId entityId);
    bool removeFluid(common::EntityId entityId);
    bool removeParticleSequence(ParticleSequenceId sequenceId);
    bool removeParticleDistanceConstraint(ParticleConstraintId constraintId);
    bool removeRigidParticleAttachmentConstraint(RigidParticleAttachmentConstraintId constraintId);
    bool removeStrandRigidAttachmentConstraint(StrandRigidAttachmentConstraintId constraintId);
    bool removeRigidDistanceConstraint(RigidDistanceConstraintId constraintId);
    bool removeRoutedCableConstraint(RoutedCableConstraintId constraintId);
    bool removeParticleCollisionFilter(ParticleCollisionFilterId filterId);
    bool removeSuturingSequence(SuturingSequenceId sequenceId);
    bool upsertBallJoint(const BallJointState &state);
    bool upsertSphericalJoint(const SphericalJointState &state);
    bool upsertHingeJoint(const HingeJointState &state);
    bool upsertSliderJoint(const SliderJointState &state);
    bool removeBallJoint(BallJointId jointId);
    bool removeSphericalJoint(SphericalJointId jointId);
    bool removeHingeJoint(HingeJointId jointId);
    bool removeSliderJoint(SliderJointId jointId);

    RigidBodyState *tryGetRigidBody(common::EntityId entityId);
    const RigidBodyState *tryGetRigidBody(common::EntityId entityId) const;
    const ColliderState *tryGetCollider(ColliderId colliderId) const;
    SoftBodyState *tryGetSoftBody(common::EntityId entityId);
    const SoftBodyState *tryGetSoftBody(common::EntityId entityId) const;
    StrandState *tryGetStrand(common::EntityId entityId);
    const StrandState *tryGetStrand(common::EntityId entityId) const;
    bool tryGetSoftBodyAuthoringRestPositions(
        common::EntityId entityId, std::vector<Diligent::float3> &outRestPositions) const;
    FluidState *tryGetFluid(common::EntityId entityId);
    const FluidState *tryGetFluid(common::EntityId entityId) const;
    AuthoredParticleSequenceState *tryGetParticleSequence(ParticleSequenceId sequenceId);
    const AuthoredParticleSequenceState *tryGetParticleSequence(
        ParticleSequenceId sequenceId) const;
    AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        ParticleConstraintId constraintId);
    const AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        ParticleConstraintId constraintId) const;
    AuthoredRigidParticleAttachmentConstraintState *tryGetRigidParticleAttachmentConstraint(
        RigidParticleAttachmentConstraintId constraintId);
    const AuthoredRigidParticleAttachmentConstraintState *tryGetRigidParticleAttachmentConstraint(
        RigidParticleAttachmentConstraintId constraintId) const;
    AuthoredStrandRigidAttachmentConstraintState *tryGetStrandRigidAttachmentConstraint(
        StrandRigidAttachmentConstraintId constraintId);
    const AuthoredStrandRigidAttachmentConstraintState *tryGetStrandRigidAttachmentConstraint(
        StrandRigidAttachmentConstraintId constraintId) const;
    AuthoredRigidDistanceConstraintState *tryGetRigidDistanceConstraint(
        RigidDistanceConstraintId constraintId);
    const AuthoredRigidDistanceConstraintState *tryGetRigidDistanceConstraint(
        RigidDistanceConstraintId constraintId) const;
    AuthoredRoutedCableConstraintState *tryGetRoutedCableConstraint(
        RoutedCableConstraintId constraintId);
    const AuthoredRoutedCableConstraintState *tryGetRoutedCableConstraint(
        RoutedCableConstraintId constraintId) const;
    AuthoredParticleCollisionFilterState *tryGetParticleCollisionFilter(
        ParticleCollisionFilterId filterId);
    const AuthoredParticleCollisionFilterState *tryGetParticleCollisionFilter(
        ParticleCollisionFilterId filterId) const;
    AuthoredSuturingSequenceState *tryGetSuturingSequence(SuturingSequenceId sequenceId);
    const AuthoredSuturingSequenceState *tryGetSuturingSequence(
        SuturingSequenceId sequenceId) const;
    const BallJointState *tryGetBallJoint(BallJointId jointId) const noexcept;
    const SphericalJointState *tryGetSphericalJoint(SphericalJointId jointId) const noexcept;
    const HingeJointState *tryGetHingeJoint(HingeJointId jointId) const noexcept;
    const SliderJointState *tryGetSliderJoint(SliderJointId jointId) const noexcept;

    const std::vector<RigidBodyState> &rigidBodySnapshot() const noexcept;
    const std::vector<ColliderState> &colliderSnapshot() const noexcept;
    const std::vector<SoftBodyState> &softBodySnapshot() const noexcept;
    const std::vector<StrandState> &strandSnapshot() const noexcept;
    const std::vector<FluidState> &fluidSnapshot() const noexcept;
    const std::vector<AuthoredParticleSequenceState> &particleSequenceSnapshot() const noexcept;
    const std::vector<AuthoredParticleDistanceConstraintState> &particleDistanceConstraintSnapshot()
        const noexcept;
    const std::vector<AuthoredRigidParticleAttachmentConstraintState> &
    rigidParticleAttachmentConstraintSnapshot() const noexcept;
    const std::vector<AuthoredStrandRigidAttachmentConstraintState> &
    strandRigidAttachmentConstraintSnapshot() const noexcept;
    const std::vector<AuthoredRigidDistanceConstraintState> &rigidDistanceConstraintSnapshot()
        const noexcept;
    const std::vector<AuthoredRoutedCableConstraintState> &routedCableConstraintSnapshot()
        const noexcept;
    const std::vector<AuthoredParticleCollisionFilterState> &particleCollisionFilterSnapshot()
        const noexcept;
    const std::vector<AuthoredSuturingSequenceState> &suturingSequenceSnapshot() const noexcept;
    const std::vector<BallJointState> &ballJointSnapshot() const noexcept;
    const std::vector<SphericalJointState> &sphericalJointSnapshot() const noexcept;
    const std::vector<HingeJointState> &hingeJointSnapshot() const noexcept;
    const std::vector<SliderJointState> &sliderJointSnapshot() const noexcept;
    const RigidBodySoAHost &rigidBodySoA() const noexcept;
    const ColliderSoAHost &colliderSoA() const noexcept;
    const BodyColliderMappingHost &bodyColliderMapping() const noexcept;
    const RigidJointSceneHost &rigidJointScene() const noexcept;
    const JointCollisionSuppressionHost &jointCollisionSuppression() const noexcept;
    const ParticleSoAHost &particles() const noexcept;
    const std::vector<Diligent::float4> &particleContactMaterials() const noexcept;
    const std::vector<FluidMaterialGpu> &fluidMaterials() const noexcept;
    const std::vector<DeformableDistanceConstraint> &distanceConstraints() const noexcept;
    const std::vector<DeformableBendConstraint> &bendConstraints() const noexcept;
    const std::vector<DeformableVolumeConstraint> &volumeConstraints() const noexcept;
    const std::vector<SoftEdge> &softEdges() const noexcept;
    bool setSoftEdgeState(std::uint32_t edgeIndex, const SoftEdge &edge) noexcept;
    bool setSoftEdgeFlags(std::uint32_t edgeIndex, std::uint32_t flags) noexcept;
    const CuttingToolGPU &cuttingTool() const noexcept;
    void setCuttingTool(const CuttingToolGPU &tool) noexcept;
    const ElectrocauteryToolGPU &electrocauteryTool() const noexcept;
    void setElectrocauteryTool(const ElectrocauteryToolGPU &tool) noexcept;
    const std::vector<SoftBend> &softBends() const noexcept;
    const std::vector<SoftTet> &softTets() const noexcept;
    const std::vector<StrandSegmentConstraint> &strandSegments() const noexcept;
    const std::vector<StrandJointConstraint> &strandJoints() const noexcept;
    const std::vector<StrandDistanceConstraint> &strandDistanceConstraints() const noexcept;
    const std::vector<StrandSegmentState> &strandSegmentStates() const noexcept;
    const std::vector<RigidParticleAttachmentConstraint> &rigidParticleAttachments() const noexcept;
    const std::vector<StrandRigidAttachmentConstraint> &strandRigidAttachments() const noexcept;
    const std::vector<RigidDistanceConstraint> &rigidDistanceConstraints() const noexcept;
    const std::vector<RoutedCableConstraint> &routedCableConstraints() const noexcept;
    const std::vector<RoutedCableRoutePoint> &routedCableRoutePoints() const noexcept;
    const std::vector<StrandSoftSuturingPair> &suturingPairs() const noexcept;
    const std::vector<std::uint32_t> &suturingParticleIndices() const noexcept;
    const SoftRenderDataHost &softRenderData() const noexcept;
    const ShapeMatchingDataHost &shapeMatchingData() const noexcept;
    void setSoftRenderData(const SoftRenderDataHost &data);
    std::uint32_t validateSoftRenderSkinningAgainstActiveEdges(
        std::vector<std::uint32_t> *outVertexComponents = nullptr) noexcept;
    const CurveRenderDataHost &curveRenderData() const noexcept;
    void setCurveRenderData(const CurveRenderDataHost &data);
    void ensureDerivedStateUpToDate() const noexcept;
    void ensureSoftBodyDerivedStateUpToDate() noexcept;
    const std::vector<std::uint32_t> &rigidBodyDirtyIndices() const noexcept;
    const std::vector<std::uint32_t> &colliderDirtyIndices() const noexcept;
    std::uint32_t rigidBodyCount() const noexcept;
    std::uint32_t colliderCount() const noexcept;
    std::uint32_t softBodyCount() const noexcept;
    std::uint32_t strandCount() const noexcept;
    std::uint32_t fluidCount() const noexcept;
    bool rigidBodyCountDirty() const noexcept;
    bool colliderCountDirty() const noexcept;
    bool fullRigidBodyUploadRequired() const noexcept;
    bool fullColliderUploadRequired() const noexcept;
    void clearRigidBodyUploadState() noexcept;
    void clearColliderUploadState() noexcept;
    bool staticBroadPhaseDirty() const noexcept;
    void clearStaticBroadPhaseDirty() noexcept;
    std::uint32_t activeMovingColliderCount() const noexcept;
    std::uint32_t staticColliderCount() const noexcept;
    float particleGridCellSize() const noexcept;
    std::uint32_t softBodyBoundsChunkCount() const noexcept;
    std::uint32_t maxSuturingPathsPerPair() const noexcept;
    std::uint32_t maxSuturingNodesPerPath() const noexcept;
    std::uint32_t suturingParticleCount() const noexcept;
    std::uint32_t reservedSuturingPathHeaderCount() const noexcept;
    std::uint32_t reservedSuturingPathNodeCount() const noexcept;

    void integrateRigidBodiesCpu(float dt) noexcept;
    bool syncRigidBodyStateFromSimulation(std::uint32_t index,
                                          const Diligent::float4 &positionInvMass,
                                          const Diligent::float4 &orientation,
                                          const Diligent::float4 &linearVelocity,
                                          const Diligent::float4 &angularVelocity) noexcept;
    void finalizeRigidBodyWriteback() noexcept;
    bool syncParticleStateFromSimulation(std::uint32_t index,
                                         const Diligent::float4 &positionInvMass,
                                         const Diligent::float4 &previousPosition,
                                         const Diligent::float4 &velocity) noexcept;
    void finalizeParticleWriteback() noexcept;
    bool syncSoftEdgeStateFromSimulation(std::uint32_t index,
                                         const SoftEdge &edge) noexcept;
    void finalizeSoftEdgeWriteback() noexcept;

    std::uint64_t authoredRevision() const noexcept;
    std::uint64_t simulationRevision() const noexcept;
    std::uint64_t rigidBodyTopologyRevision() const noexcept;
    std::uint64_t rigidJointTopologyRevision() const noexcept;
    std::uint64_t rigidJointSceneRevision() const noexcept;
    std::uint64_t rigidJointModeRevision() const noexcept;
    std::uint64_t softBodyTopologyRevision() const noexcept;
    std::uint64_t softParticleRevision() const noexcept;
    std::uint64_t softTopologyRevision() const noexcept;
    std::uint64_t softConstraintAdjacencyRevision() const noexcept;
    std::uint64_t rigidParticleAttachmentDefinitionRevision() const noexcept;
    std::uint64_t rigidParticleAttachmentResolvedRevision() const noexcept;
    std::uint64_t strandRigidAttachmentDefinitionRevision() const noexcept;
    std::uint64_t strandRigidAttachmentResolvedRevision() const noexcept;
    std::uint64_t rigidDistanceConstraintDefinitionRevision() const noexcept;
    std::uint64_t rigidDistanceConstraintResolvedRevision() const noexcept;
    std::uint64_t routedCableDefinitionRevision() const noexcept;
    std::uint64_t routedCableResolvedRevision() const noexcept;
    std::uint64_t curveRenderRevision() const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
