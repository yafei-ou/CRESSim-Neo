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
        state.kinematicTargetPosition = rigidBody->kinematicTargetPosition;
        state.kinematicTargetRotation = rigidBody->kinematicTargetRotation;
        state.kinematicTargetEnabled  = rigidBody->kinematicTargetEnabled;
        const physics::RigidBodyState& persistedBody = physicsWorld.upsertRigidBody(state);

        std::vector<physics::ColliderState> colliders;
        const auto& colliderHandles = world.colliderHandles(entityId);
        colliders.reserve(colliderHandles.size());
        for (const ColliderHandle handle : colliderHandles)
        {
            const ColliderComponent* collider = world.tryGetCollider(handle);
            if (collider == nullptr)
            {
                continue;
            }

            physics::ColliderState colliderState{};
            colliderState.entityId         = entityId;
            colliderState.ownerRigidBodyId = persistedBody.rigidBodyId;
            colliderState.shapeType        = collider->shapeType;
            colliderState.shapeParams      = collider->shapeParams;
            colliderState.localPosition    = collider->localPosition;
            colliderState.localRotation    = collider->localRotation;
            colliderState.enabled          = collider->enabled;
            colliderState.friction         = collider->friction;
            colliderState.restitution      = collider->restitution;
            colliderState.collisionLayer   = collider->collisionLayer;
            colliderState.collisionMask    = collider->collisionMask;
            colliders.push_back(colliderState);
        }

        physicsWorld.replaceColliders(entityId, colliders);
    }
}

} // namespace cressim::neo::engine::detail
