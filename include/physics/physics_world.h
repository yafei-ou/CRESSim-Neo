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

    RigidBodyState& upsertRigidBody(const RigidBodyState& state);
    bool removeRigidBody(common::EntityId entityId);
    void upsertCollider(const ColliderState& collider);
    bool removeCollider(ColliderId colliderId);
    void replaceColliders(common::EntityId entityId, const std::vector<ColliderState>& colliders);

    RigidBodyState* tryGetRigidBody(common::EntityId entityId);
    const RigidBodyState* tryGetRigidBody(common::EntityId entityId) const;
    const ColliderState* tryGetCollider(ColliderId colliderId) const;

    const std::vector<RigidBodyState>& rigidBodySnapshot() const noexcept;
    const std::vector<ColliderState>& colliderSnapshot() const noexcept;
    const RigidBodySoAHost& rigidBodySoA() const noexcept;
    const ColliderSoAHost& colliderSoA() const noexcept;
    const BodyColliderMappingHost& bodyColliderMapping() const noexcept;
    const PhysicsSoADirtyRange& rigidBodyDirtyRange() const noexcept;
    const PhysicsSoADirtyRange& colliderDirtyRange() const noexcept;
    std::uint32_t rigidBodyCount() const noexcept;
    std::uint32_t colliderCount() const noexcept;
    void clearRigidBodyDirtyRange() noexcept;
    void clearColliderDirtyRange() noexcept;
    bool staticBroadPhaseDirty() const noexcept;
    void clearStaticBroadPhaseDirty() noexcept;

    void integrateRigidBodiesCpu(float dt) noexcept;
    bool writeBackRigidBodyState(std::uint32_t index, const Diligent::float4& positionInvMass,
                                 const Diligent::float4& orientation,
                                 const Diligent::float4& linearVelocity,
                                 const Diligent::float4& angularVelocity) noexcept;
    void finalizeRigidBodyWriteback() noexcept;

    std::uint64_t revision() const noexcept;

private:
    static void writeRigidBodySoAAt(RigidBodySoAHost& soa, std::uint32_t index,
                                    const RigidBodyState& state);
    static void writeColliderSoAAt(ColliderSoAHost& soa, std::uint32_t index,
                                   const ColliderState& state, std::uint32_t ownerBodyIndex);
    static bool isStaticBody(const RigidBodyState& state) noexcept;
    static bool staticBodyPoseChanged(const RigidBodyState& before,
                                      const RigidBodyState& after) noexcept;
    static void normalizeRigidBodyState(RigidBodyState& state) noexcept;
    static void normalizeColliderState(ColliderState& state) noexcept;

    void removeCollidersForEntity(common::EntityId entityId) noexcept;
    void removeColliderAtIndex(std::uint32_t index) noexcept;
    void rebuildBodyColliderMapping() noexcept;
    void markAllRigidBodiesDirty() noexcept;
    void markAllCollidersDirty() noexcept;

    RigidBodySoAHost mRigidBodies{};
    ColliderSoAHost mColliders{};
    BodyColliderMappingHost mBodyColliderMapping{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToRigidBodyIndex{};
    std::unordered_map<RigidBodyId, std::uint32_t> mRigidBodyIdToIndex{};
    std::unordered_map<ColliderId, std::uint32_t> mColliderIdToIndex{};
    std::unordered_map<common::EntityId, std::vector<ColliderId>> mEntityToColliderIds{};
    std::vector<RigidBodyState> mRigidBodySnapshot{};
    std::vector<ColliderState> mColliderSnapshot{};
    PhysicsSoADirtyRange mRigidBodyDirtyRange{};
    PhysicsSoADirtyRange mColliderDirtyRange{};
    bool mStaticBroadPhaseDirty  = false;
    std::uint64_t mRevision      = 0;
    RigidBodyId mNextRigidBodyId = 1u;
    ColliderId mNextColliderId   = 1u;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
