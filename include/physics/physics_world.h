#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H

#include "physics/export.h"
#include "physics/physics_types.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace cressim::neo::physics
{

class CRESSIM_NEO_PHYSICS_API PhysicsWorld
{
public:
    void clear();

    RigidBodyState &upsertRigidBody(const RigidBodyState &state);
    bool removeRigidBody(common::EntityId entityId);
    void upsertCollider(const ColliderState &collider);
    bool removeCollider(ColliderId colliderId);
    void replaceColliders(common::EntityId entityId, const std::vector<ColliderState> &colliders);
    bool upsertSoftBody(const SoftBodyState &state);
    bool upsertStrand(const StrandState &state);
    bool upsertFluid(const FluidState &state);
    AuthoredParticleDistanceConstraintState &upsertParticleDistanceConstraint(
        const AuthoredParticleDistanceConstraintState &state);
    AuthoredSuturingSequenceState &upsertSuturingSequence(
        const AuthoredSuturingSequenceState &state);
    bool removeSoftBody(common::EntityId entityId);
    bool removeStrand(common::EntityId entityId);
    bool removeFluid(common::EntityId entityId);
    bool removeParticleDistanceConstraint(ParticleConstraintId constraintId);
    bool removeSuturingSequence(SuturingSequenceId sequenceId);
    bool upsertBallJoint(const BallJointState &state);
    bool upsertHingeJoint(const HingeJointState &state);
    bool upsertSliderJoint(const SliderJointState &state);
    bool removeBallJoint(BallJointId jointId);
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
    AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        ParticleConstraintId constraintId);
    const AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        ParticleConstraintId constraintId) const;
    AuthoredSuturingSequenceState *tryGetSuturingSequence(SuturingSequenceId sequenceId);
    const AuthoredSuturingSequenceState *tryGetSuturingSequence(
        SuturingSequenceId sequenceId) const;
    const BallJointState *tryGetBallJoint(BallJointId jointId) const noexcept;
    const HingeJointState *tryGetHingeJoint(HingeJointId jointId) const noexcept;
    const SliderJointState *tryGetSliderJoint(SliderJointId jointId) const noexcept;

    const std::vector<RigidBodyState> &rigidBodySnapshot() const noexcept;
    const std::vector<ColliderState> &colliderSnapshot() const noexcept;
    const std::vector<SoftBodyState> &softBodySnapshot() const noexcept;
    const std::vector<StrandState> &strandSnapshot() const noexcept;
    const std::vector<FluidState> &fluidSnapshot() const noexcept;
    const std::vector<AuthoredParticleDistanceConstraintState> &
    particleDistanceConstraintSnapshot() const noexcept;
    const std::vector<AuthoredSuturingSequenceState> &suturingSequenceSnapshot() const noexcept;
    const std::vector<BallJointState> &ballJointSnapshot() const noexcept;
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
    const std::vector<SoftBend> &softBends() const noexcept;
    const std::vector<SoftTet> &softTets() const noexcept;
    const std::vector<StrandSoftSuturingPair> &suturingPairs() const noexcept;
    const std::vector<std::uint32_t> &suturingParticleIndices() const noexcept;
    const SoftRenderDataHost &softRenderData() const noexcept;
    void setSoftRenderData(const SoftRenderDataHost &data);
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
    void syncSuturingStateFromSimulation(
        const std::vector<GpuStrandInsertionState> &insertionStates,
        const std::vector<SuturingPathHeader> &pathHeaders,
        const std::vector<SuturingPathNode> &pathNodes) noexcept;
    bool overrideStrandParticlePosition(common::EntityId entityId, std::uint32_t localParticleIndex,
                                        const Diligent::float3 &position,
                                        bool updatePreviousPosition = true) noexcept;
    void finalizeParticleWriteback() noexcept;

    std::uint64_t authoredRevision() const noexcept;
    std::uint64_t simulationRevision() const noexcept;
    std::uint64_t rigidBodyTopologyRevision() const noexcept;
    std::uint64_t rigidJointTopologyRevision() const noexcept;
    std::uint64_t rigidJointSceneRevision() const noexcept;
    std::uint64_t rigidJointModeRevision() const noexcept;
    std::uint64_t softBodyTopologyRevision() const noexcept;
    std::uint64_t softParticleRevision() const noexcept;
    std::uint64_t softGpuTopologyRevision() const noexcept;

private:
    enum class SoftBodyChangeKind
    {
        RuntimePropertiesOnly,
        TopologyRebuild,
    };

    enum class StrandChangeKind
    {
        RuntimePropertiesOnly,
        TopologyRebuild,
    };

    enum class RigidJointChangeKind
    {
        PayloadOnly,
        ModeRebuild,
        TopologyRebuild,
    };

    struct SoftBodyDerivedCache
    {
        std::vector<Diligent::float3> restPositions;
        std::vector<std::array<std::uint32_t, 2>> edges;
        std::vector<std::array<std::uint32_t, 4>> tets;
        std::vector<Diligent::uint3> boundaryFaces;
        std::vector<std::vector<std::uint32_t>> adjacencyLists;
        std::vector<std::vector<std::uint32_t>> incidentTetLists;
        std::vector<std::uint32_t> staticParticleIndices;
    };

    struct FluidDerivedCache
    {
        std::vector<Diligent::float3> restPositions;
    };

    struct StrandDerivedCache
    {
        std::vector<Diligent::float3> restPositions;
        std::vector<std::array<std::uint32_t, 2>> edges;
        std::vector<std::array<std::uint32_t, 3>> bends;
        std::vector<std::vector<std::uint32_t>> adjacencyLists;
        std::vector<std::uint32_t> staticParticleIndices;
    };

    struct StrandParticleInsertionState
    {
        StrandInsertionState state       = StrandInsertionState::Outside;
        StrandInsertionState previous    = StrandInsertionState::Outside;
        std::uint32_t softBodyIndex      = 0xffffffffu;
        std::uint32_t tetIndex           = 0xffffffffu;
        Diligent::float4 barycentrics{0.0f, 0.0f, 0.0f, 0.0f};
        std::uint32_t nearestPathNode    = 0xffffffffu;
    };

    struct StrandSuturingState
    {
        std::vector<StrandParticleInsertionState> particleStates;
        std::vector<SuturingPathNode> pathNodes;
        bool pathActive              = false;
        std::uint32_t activeSoftBody = 0xffffffffu;
    };

    static void writeRigidBodySoAAt(RigidBodySoAHost &soa, std::uint32_t index,
                                    const RigidBodyState &state);
    static void writeColliderSoAAt(ColliderSoAHost &soa, std::uint32_t index,
                                   const ColliderState &state, std::uint32_t ownerBodyIndex,
                                   std::uint32_t ownerEnvironmentIndex);
    static bool isStaticBody(const RigidBodyState &state) noexcept;
    static bool staticBodyPoseChanged(const RigidBodyState &before,
                                      const RigidBodyState &after) noexcept;
    static bool isMovingBody(const RigidBodyState &state) noexcept;
    static void normalizeRigidBodyState(RigidBodyState &state) noexcept;
    static void normalizeColliderState(ColliderState &state) noexcept;
    static std::uint32_t colliderBroadPhaseContribution(const ColliderState &collider,
                                                        const RigidBodyState *owner) noexcept;

    void markRigidBodyDirty(std::uint32_t index) noexcept;
    void markColliderDirty(std::uint32_t index) noexcept;
    void markRigidBodyCountDirty(bool fullUploadRequired = false) noexcept;
    void markColliderCountDirty(bool fullUploadRequired = false) noexcept;
    void markJointSceneDirty() noexcept;
    void markJointModeDirty() noexcept;
    void markJointTopologyDirty() noexcept;
    void removeCollidersForEntity(common::EntityId entityId) noexcept;
    void removeColliderAtIndex(std::uint32_t index) noexcept;
    bool pruneRigidJointsForBody(RigidBodyId rigidBodyId) noexcept;
    void rebuildBodyColliderMapping() const noexcept;
    void rebuildRigidJointScene() const noexcept;
    void rebuildJointCollisionSuppression() const noexcept;
    void rebuildSoftBodyDerivedState() noexcept;
    void markAllRigidBodiesDirty() noexcept;
    void markAllCollidersDirty() noexcept;
    std::uint32_t broadPhaseContributionForCollider(const ColliderState &collider) const noexcept;
    std::uint32_t enabledColliderCountForEntity(common::EntityId entityId) const noexcept;
    static void normalizeSoftBodyState(SoftBodyState &state) noexcept;
    static void normalizeStrandState(StrandState &state) noexcept;
    static void normalizeFluidState(FluidState &state) noexcept;
    bool validateFluidMaterialCompatibility(const FluidState &candidate,
                                            const FluidState *previousState) const noexcept;
    static SoftBodyChangeKind classifySoftBodyChange(const SoftBodyState &previousState,
                                                     const SoftBodyState &candidate) noexcept;
    static StrandChangeKind classifyStrandChange(const StrandState &previousState,
                                                 const StrandState &candidate) noexcept;
    static RigidJointChangeKind classifyBallJointChange(bool inserted) noexcept;
    static RigidJointChangeKind classifyHingeJointChange(const HingeJointState &previousState,
                                                         const HingeJointState &candidate,
                                                         bool inserted) noexcept;
    static RigidJointChangeKind classifySliderJointChange(const SliderJointState &previousState,
                                                          const SliderJointState &candidate,
                                                          bool inserted) noexcept;
    void applyRigidJointChange(RigidJointChangeKind changeKind) noexcept;
    void applySoftBodyRuntimeProperties(std::uint32_t index,
                                        const SoftBodyState &normalizedState) noexcept;
    void applyStrandRuntimeProperties(std::uint32_t index,
                                      const StrandState &normalizedState) noexcept;
    void recomputeParticleGridCellSize() noexcept;
    void recomputeSoftBodyBoundsChunkCount() noexcept;
    bool prepareSoftBodyStateForInsert(const SoftBodyState &candidate,
                                       const SoftBodyState *previousState,
                                       SoftBodyDerivedCache &derivedCache) noexcept;
    bool prepareStrandStateForInsert(const StrandState &candidate,
                                     StrandDerivedCache &derivedCache) noexcept;
    bool prepareFluidStateForInsert(const FluidState &candidate,
                                    FluidDerivedCache &derivedCache) noexcept;
    struct TetGenMeshCache
    {
        std::string nodeFile;
        std::string eleFile;
        std::vector<Diligent::float3> objectSpaceRestPositions;
        std::vector<std::uint32_t> tetVertexIndices;
    };

    const TetGenMeshCache *tryGetTetGenMeshCache(common::EntityId entityId) const noexcept;

    RigidBodySoAHost mRigidBodies{};
    mutable ColliderSoAHost mColliders{};
    mutable BodyColliderMappingHost mBodyColliderMapping{};
    mutable RigidJointSceneHost mRigidJointScene{};
    mutable JointCollisionSuppressionHost mJointCollisionSuppression{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToRigidBodyIndex{};
    std::unordered_map<RigidBodyId, std::uint32_t> mRigidBodyIdToIndex{};
    std::unordered_map<ColliderId, std::uint32_t> mColliderIdToIndex{};
    std::unordered_map<common::EntityId, std::vector<ColliderId>> mEntityToColliderIds{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToSoftBodyIndex{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToStrandIndex{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToFluidIndex{};
    std::unordered_map<ParticleConstraintId, std::uint32_t> mParticleConstraintIdToIndex{};
    std::unordered_map<SuturingSequenceId, std::uint32_t> mSuturingSequenceIdToIndex{};
    std::unordered_map<common::EntityId, TetGenMeshCache> mTetGenMeshCache{};
    std::vector<RigidBodyState> mRigidBodySnapshot{};
    std::vector<ColliderState> mColliderSnapshot{};
    std::vector<SoftBodyState> mSoftBodySnapshot{};
    std::vector<StrandState> mStrandSnapshot{};
    std::vector<FluidState> mFluidSnapshot{};
    std::vector<AuthoredParticleDistanceConstraintState> mParticleDistanceConstraintSnapshot{};
    std::vector<AuthoredSuturingSequenceState> mSuturingSequenceSnapshot{};
    std::vector<BallJointState> mBallJointSnapshot{};
    std::vector<HingeJointState> mHingeJointSnapshot{};
    std::vector<SliderJointState> mSliderJointSnapshot{};
    std::vector<SoftBodyDerivedCache> mSoftBodyDerivedCaches{};
    std::vector<StrandDerivedCache> mStrandDerivedCaches{};
    std::vector<FluidDerivedCache> mFluidDerivedCaches{};
    std::vector<StrandSuturingState> mStrandSuturingStates{};
    std::vector<StrandSoftSuturingPair> mSuturingPairs{};
    ParticleSoAHost mParticles{};
    std::vector<Diligent::float4> mParticleContactMaterials{};
    std::vector<FluidMaterialGpu> mFluidMaterials{};
    std::vector<DeformableDistanceConstraint> mSoftEdges{};
    std::vector<DeformableBendConstraint> mSoftBends{};
    std::vector<DeformableVolumeConstraint> mSoftTets{};
    SoftRenderDataHost mSoftRenderData{};
    std::vector<std::uint32_t> mRigidBodyDirtyIndices{};
    std::vector<std::uint32_t> mColliderDirtyIndices{};
    std::vector<std::uint8_t> mRigidBodyDirtyBits{};
    std::vector<std::uint8_t> mColliderDirtyBits{};
    bool mRigidBodyCountDirty                    = false;
    bool mColliderCountDirty                     = false;
    bool mFullRigidBodyUploadRequired            = false;
    bool mFullColliderUploadRequired             = false;
    mutable bool mBodyColliderMappingDirty       = true;
    mutable bool mRigidJointSceneDirty           = true;
    mutable bool mJointCollisionSuppressionDirty = true;
    bool mSoftBodyDerivedStateDirty              = true;
    bool mStaticBroadPhaseDirty                  = false;
    std::uint32_t mActiveMovingColliderCount     = 0u;
    std::uint32_t mStaticColliderCount           = 0u;
    float mParticleGridCellSize                  = 0.1f;
    std::uint32_t mSoftBodyBoundsChunkCount      = 0u;
    std::uint32_t mMaxSuturingPathsPerPair       = 4u;
    std::uint32_t mMaxSuturingNodesPerPath       = 128u;
    std::uint32_t mReservedSuturingPathHeaders   = 0u;
    std::uint32_t mReservedSuturingPathNodes     = 0u;
    std::uint64_t mAuthoredRevision              = 0;
    std::uint64_t mSimulationRevision            = 0;
    std::uint64_t mRigidBodyTopologyRevision     = 0;
    std::uint64_t mRigidJointSceneRevision       = 0;
    std::uint64_t mRigidJointModeRevision        = 0;
    std::uint64_t mRigidJointTopologyRevision    = 0;
    std::uint64_t mSoftBodyTopologyRevision      = 0;
    std::uint64_t mSoftParticleRevision          = 0;
    std::uint64_t mSoftGpuTopologyRevision       = 0;
    RigidBodyId mNextRigidBodyId                 = 1u;
    ColliderId mNextColliderId                   = 1u;
    BallJointId mNextBallJointId                 = 1u;
    HingeJointId mNextHingeJointId               = 1u;
    SliderJointId mNextSliderJointId             = 1u;
    ParticleConstraintId mNextParticleConstraintId = 1u;
    SuturingSequenceId mNextSuturingSequenceId     = 1u;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
