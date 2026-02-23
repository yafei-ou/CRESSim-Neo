#include "engine/physics_world_to_world_sync.h"

namespace cressim::neo::engine::detail
{

void syncPhysicsWorldToWorld(const physics::PhysicsWorld& physicsWorld, World& world)
{
    for (const physics::RigidBodyState& state : physicsWorld.rigidBodySnapshot())
    {
        if (!world.isAlive(state.entityId))
        {
            continue;
        }

        TransformComponent transform{};
        const TransformComponent* existing = world.tryGetTransform(state.entityId);
        if (existing != nullptr)
        {
            transform = *existing;
        }

        transform.worldTransform.position = state.position;
        transform.worldTransform.rotation = state.rotation;
        (void)world.setTransform(state.entityId, transform);
    }
}

} // namespace cressim::neo::engine::detail
