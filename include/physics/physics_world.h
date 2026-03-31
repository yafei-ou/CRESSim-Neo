#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H

#include "physics/export.h"
#include "physics/physics_types.h"

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

    RigidBodyState *tryGetRigidBody(common::EntityId entityId);
    const RigidBodyState *tryGetRigidBody(common::EntityId entityId) const;
    const ColliderState *tryGetCollider(ColliderId colliderId) const;

    const std::vector<RigidBodyState> &rigidBodySnapshot() const noexcept;
    const std::vector<ColliderState> &colliderSnapshot() const noexcept;
    const RigidBodySoAHost &rigidBodySoA() const noexcept;
    const ColliderSoAHost &colliderSoA() const noexcept;
    const BodyColliderMappingHost &bodyColliderMapping() const noexcept;
    void ensureDerivedStateUpToDate() const noexcept;
    const std::vector<std::uint32_t> &rigidBodyDirtyIndices() const noexcept;
    const std::vector<std::uint32_t> &colliderDirtyIndices() const noexcept;
    std::uint32_t rigidBodyCount() const noexcept;
    std::uint32_t colliderCount() const noexcept;
    bool rigidBodyCountDirty() const noexcept;
    bool colliderCountDirty() const noexcept;
    bool fullRigidBodyUploadRequired() const noexcept;
    bool fullColliderUploadRequired() const noexcept;
    void clearRigidBodyUploadState() noexcept;
    void clearColliderUploadState() noexcept;
    bool staticBroadPhaseDirty() const noexcept;
    void clearStaticBroadPhaseDirty() noexcept;

    void integrateRigidBodiesCpu(float dt) noexcept;
    bool writeBackRigidBodyState(std::uint32_t index, const Diligent::float4 &positionInvMass,
                                 const Diligent::float4 &orientation,
                                 const Diligent::float4 &linearVelocity,
                                 const Diligent::float4 &angularVelocity) noexcept;
    void finalizeRigidBodyWriteback() noexcept;

    std::uint64_t revision() const noexcept;
    std::uint64_t rigidBodyTopologyRevision() const noexcept;

private:
    static void writeRigidBodySoAAt(RigidBodySoAHost &soa, std::uint32_t index,
                                    const RigidBodyState &state);
    static void writeColliderSoAAt(ColliderSoAHost &soa, std::uint32_t index,
                                   const ColliderState &state, std::uint32_t ownerBodyIndex);
    static bool isStaticBody(const RigidBodyState &state) noexcept;
    static bool staticBodyPoseChanged(const RigidBodyState &before,
                                      const RigidBodyState &after) noexcept;
    static void normalizeRigidBodyState(RigidBodyState &state) noexcept;
    static void normalizeColliderState(ColliderState &state) noexcept;

    void markRigidBodyDirty(std::uint32_t index) noexcept;
    void markColliderDirty(std::uint32_t index) noexcept;
    void markRigidBodyCountDirty(bool fullUploadRequired = false) noexcept;
    void markColliderCountDirty(bool fullUploadRequired = false) noexcept;
    void removeCollidersForEntity(common::EntityId entityId) noexcept;
    void removeColliderAtIndex(std::uint32_t index) noexcept;
    void rebuildBodyColliderMapping() const noexcept;
    void markAllRigidBodiesDirty() noexcept;
    void markAllCollidersDirty() noexcept;

    RigidBodySoAHost mRigidBodies{};
    mutable ColliderSoAHost mColliders{};
    mutable BodyColliderMappingHost mBodyColliderMapping{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToRigidBodyIndex{};
    std::unordered_map<RigidBodyId, std::uint32_t> mRigidBodyIdToIndex{};
    std::unordered_map<ColliderId, std::uint32_t> mColliderIdToIndex{};
    std::unordered_map<common::EntityId, std::vector<ColliderId>> mEntityToColliderIds{};
    std::vector<RigidBodyState> mRigidBodySnapshot{};
    std::vector<ColliderState> mColliderSnapshot{};
    std::vector<std::uint32_t> mRigidBodyDirtyIndices{};
    std::vector<std::uint32_t> mColliderDirtyIndices{};
    std::vector<std::uint8_t> mRigidBodyDirtyBits{};
    std::vector<std::uint8_t> mColliderDirtyBits{};
    bool mRigidBodyCountDirty                = false;
    bool mColliderCountDirty                 = false;
    bool mFullRigidBodyUploadRequired        = false;
    bool mFullColliderUploadRequired         = false;
    mutable bool mBodyColliderMappingDirty   = true;
    bool mStaticBroadPhaseDirty              = false;
    std::uint64_t mRevision                  = 0;
    std::uint64_t mRigidBodyTopologyRevision = 0;
    RigidBodyId mNextRigidBodyId             = 1u;
    ColliderId mNextColliderId               = 1u;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
