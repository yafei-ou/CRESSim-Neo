#include "engine/world_to_physics_world_sync.h"

namespace cressim::neo::engine::detail
{

void syncWorldToPhysicsWorld(const World& world, physics::PhysicsWorld& physicsWorld)
{
    for (const common::EntityId entityId : world.dirtyEntities())
    {
        if (!world.isAlive(entityId))
        {
            (void)physicsWorld.removeRigidBody(entityId);
            continue;
        }

        const TransformComponent* transform = world.tryGetTransform(entityId);
        const RigidBodyComponent* rigidBody = world.tryGetRigidBody(entityId);
        if (transform == nullptr || rigidBody == nullptr || !rigidBody->simulated)
        {
            (void)physicsWorld.removeRigidBody(entityId);
            continue;
        }

        physics::RigidBodyState state{};
        state.entityId            = entityId;
        state.position            = transform->worldTransform.position;
        state.rotation            = transform->worldTransform.rotation;
        state.scale               = transform->worldTransform.scale;
        state.linearVelocity      = rigidBody->linearVelocity;
        state.angularVelocity     = rigidBody->angularVelocity;
        state.inverseInertiaLocal = rigidBody->inverseInertiaLocal;
        state.bodyType            = rigidBody->bodyType;
        state.inverseMass         = rigidBody->inverseMass;
        state.colliderShape       = static_cast<physics::ColliderShapeType>(
            static_cast<std::uint32_t>(rigidBody->colliderShape));
        state.colliderParams           = rigidBody->colliderParams;
        state.kinematicTargetPosition  = rigidBody->kinematicTargetPosition;
        state.kinematicTargetRotation  = rigidBody->kinematicTargetRotation;
        state.kinematicTargetEnabled   = rigidBody->kinematicTargetEnabled;
        (void)physicsWorld.upsertRigidBody(state);
    }
}

} // namespace cressim::neo::engine::detail
