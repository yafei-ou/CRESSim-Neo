#include "engine/components.h"
#include "engine/world.h"

#include <iostream>

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId entity = world.createEntity();

    engine::TransformComponent transform{};
    transform.worldTransform.position = {1.0f, 2.0f, 3.0f};
    world.setTransform(entity, transform);

    engine::RigidBodyComponent rigidBody{};
    rigidBody.bodyType = physics::RigidBodyType::Dynamic;
    rigidBody.simulated = true;
    rigidBody.linearVelocity = {0.5f, 0.0f, 0.0f};
    world.setRigidBody(entity, rigidBody);

    engine::ColliderComponent collider{};
    collider.shapeType = physics::ColliderShapeType::Box;
    const engine::ColliderHandle handle = world.addCollider(entity, collider);

    const physics::PhysicsWorld& physicsWorld = world.physicsWorld();
    if (physicsWorld.colliderCount() != 1u)
    {
        std::cerr << "Expected one collider after direct authoring.\n";
        return 1;
    }

    const physics::ColliderState* initialCollider = physicsWorld.tryGetCollider(handle.id);
    if (initialCollider == nullptr || initialCollider->colliderId != handle.id)
    {
        std::cerr << "World collider handle was not preserved in physics storage.\n";
        return 1;
    }

    world.physicsWorld().clearStaticBroadPhaseDirty();

    transform.worldTransform.position = {4.0f, 5.0f, 6.0f};
    world.setTransform(entity, transform);

    const physics::ColliderState* movedCollider = physicsWorld.tryGetCollider(handle.id);
    const physics::RigidBodyState* movedBody = physicsWorld.tryGetRigidBody(entity);
    if (movedCollider == nullptr || movedBody == nullptr ||
        movedCollider->colliderId != handle.id || movedBody->position.x != 4.0f)
    {
        std::cerr << "Dynamic transform update did not preserve collider/body identity.\n";
        return 1;
    }
    if (physicsWorld.staticBroadPhaseDirty())
    {
        std::cerr << "Dynamic transform update dirtied the static broad phase.\n";
        return 1;
    }

    world.physicsWorld().clearStaticBroadPhaseDirty();

    rigidBody.linearVelocity = {3.0f, 0.0f, 0.0f};
    world.setRigidBody(entity, rigidBody);

    const physics::ColliderState* updatedCollider = physicsWorld.tryGetCollider(handle.id);
    const physics::RigidBodyState* updatedBody = physicsWorld.tryGetRigidBody(entity);
    if (updatedCollider == nullptr || updatedBody == nullptr ||
        updatedCollider->colliderId != handle.id || updatedBody->linearVelocity.x != 3.0f)
    {
        std::cerr << "Rigid body update recreated collider state unexpectedly.\n";
        return 1;
    }
    if (physicsWorld.staticBroadPhaseDirty())
    {
        std::cerr << "Dynamic rigid body update dirtied the static broad phase.\n";
        return 1;
    }

    std::cout << "Direct physics authoring checks passed.\n";
    return 0;
}
