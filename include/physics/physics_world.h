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

    // Compatibility facade for legacy callers. Prefer rigidBodySnapshot()/rigidBodySoA().
    RigidBodyState* tryGetRigidBody(common::EntityId entityId);
    const RigidBodyState* tryGetRigidBody(common::EntityId entityId) const;

    const std::vector<RigidBodyState>& rigidBodySnapshot() const noexcept;
    const RigidBodySoAHost& rigidBodySoA() const noexcept;
    const PhysicsSoADirtyRange& rigidBodyDirtyRange() const noexcept;
    std::uint32_t rigidBodyCount() const noexcept;
    void clearRigidBodyDirtyRange() noexcept;
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
    static RigidBodyState readRigidBodySoAAt(const RigidBodySoAHost& soa, std::uint32_t index);
    static bool isStaticBody(const RigidBodyState& state) noexcept;
    static bool staticShapeChanged(const RigidBodyState& before,
                                   const RigidBodyState& after) noexcept;
    static void normalizeRigidBodyState(RigidBodyState& state) noexcept;
    void markAllRigidBodiesDirty() noexcept;

    RigidBodySoAHost mRigidBodies{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityToIndex{};
    std::vector<RigidBodyState> mRigidBodySnapshot{};
    PhysicsSoADirtyRange mRigidBodyDirtyRange{};
    bool mStaticBroadPhaseDirty = false;
    std::uint64_t mRevision = 0;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
