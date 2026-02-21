#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H

#include "physics/export.h"
#include "physics/physics_types.h"

#include <cstdint>

namespace cressim::neo::physics
{

class CRESSIM_NEO_PHYSICS_API PhysicsWorld
{
public:
    void clear();

    RigidBodyState& upsertRigidBody(const RigidBodyState& state);
    bool removeRigidBody(common::EntityId entityId);

    RigidBodyState* tryGetRigidBody(common::EntityId entityId);
    const RigidBodyState* tryGetRigidBody(common::EntityId entityId) const;

    const std::vector<RigidBodyState>& rigidBodies() const noexcept;
    std::vector<RigidBodyState>& rigidBodies() noexcept;

    std::uint64_t revision() const noexcept;

private:
    std::vector<RigidBodyState> mRigidBodies;
    std::uint64_t mRevision = 0;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_WORLD_H
