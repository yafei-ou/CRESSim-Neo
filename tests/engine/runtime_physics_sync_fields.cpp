#include "engine/components.h"
#include "engine/physics_world_to_world_sync.h"
#include "engine/world.h"
#include "engine/world_to_physics_world_sync.h"
#include "physics/physics_world.h"

#include <cstdint>
#include <cmath>
#include <iostream>

int main()
{
    using namespace cressim::neo;

    engine::World world;
    physics::PhysicsWorld physicsWorld;

    const common::EntityId entity = world.createEntity();
    engine::TransformComponent transform{};
    transform.worldTransform.position = {1.0f, 2.0f, 3.0f};
    transform.worldTransform.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    world.setTransform(entity, transform);

    engine::RigidBodyComponent rigidBody{};
    rigidBody.simulated = true;
    rigidBody.inverseMass = 0.5f;
    rigidBody.linearVelocity = {2.0f, 0.0f, -1.0f};
    rigidBody.angularVelocity = {0.0f, 3.0f, 0.0f};
    rigidBody.colliderShape = physics::ColliderShapeType::Capsule;
    rigidBody.colliderParams = {0.7f, 1.4f, 0.0f, 0.0f};
    world.setRigidBody(entity, rigidBody);

    engine::detail::syncWorldToPhysicsWorld(world, physicsWorld);
    const physics::RigidBodyState* state = physicsWorld.tryGetRigidBody(entity);
    if (state == nullptr)
    {
        std::cerr << "Rigid body missing in physics world.\n";
        return 1;
    }
    if (state->angularVelocity.y != rigidBody.angularVelocity.y ||
        static_cast<std::uint32_t>(state->colliderShape) !=
            static_cast<std::uint32_t>(physics::ColliderShapeType::Capsule) ||
        state->colliderParams.x != rigidBody.colliderParams.x)
    {
        std::cerr << "World->physics sync did not preserve new rigid fields.\n";
        return 1;
    }

    const Diligent::float4 writebackPos{4.0f, 5.0f, 6.0f, state->inverseMass};
    const Diligent::float4 writebackRot{0.0f, 0.0f, 0.3826834f, 0.9238795f};
    const Diligent::float4 writebackLin{state->linearVelocity.x, state->linearVelocity.y,
                                        state->linearVelocity.z, 0.0f};
    const Diligent::float4 writebackAng{state->angularVelocity.x, state->angularVelocity.y,
                                        state->angularVelocity.z, 0.0f};
    if (!physicsWorld.writeBackRigidBodyState(0u, writebackPos, writebackRot, writebackLin,
                                              writebackAng))
    {
        std::cerr << "Failed to write back rigid state into physics world.\n";
        return 1;
    }
    physicsWorld.finalizeRigidBodyWriteback();

    engine::detail::syncPhysicsWorldToWorld(physicsWorld, world);
    const engine::TransformComponent* syncedTransform = world.tryGetTransform(entity);
    if (syncedTransform == nullptr)
    {
        std::cerr << "Physics->world sync removed transform unexpectedly.\n";
        return 1;
    }

    const float dx = std::fabs(syncedTransform->worldTransform.position.x - writebackPos.x);
    const float dy = std::fabs(syncedTransform->worldTransform.position.y - writebackPos.y);
    const float dz = std::fabs(syncedTransform->worldTransform.position.z - writebackPos.z);
    if (dx > 1e-5f || dy > 1e-5f || dz > 1e-5f)
    {
        std::cerr << "Physics->world transform position sync mismatch.\n";
        return 1;
    }
    const float dr = std::fabs(syncedTransform->worldTransform.rotation.q.z - writebackRot.z);
    if (dr > 1e-5f)
    {
        std::cerr << "Physics->world transform rotation sync mismatch.\n";
        return 1;
    }

    std::cout << "Runtime physics sync field checks passed.\n";
    return 0;
}
