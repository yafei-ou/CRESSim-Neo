#include "physics/physics_world.h"

#include <cstdint>
#include <iostream>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::RigidBodyState;

RigidBodyState makeRigidBody(EntityId entityId, float x, float vx, ColliderShapeType shape)
{
    RigidBodyState state{};
    state.entityId = entityId;
    state.position = {x, 0.0f, 0.0f};
    state.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.scale = {1.0f + x, 2.0f + x, 3.0f + x};
    state.linearVelocity = {vx, 0.0f, 0.0f};
    state.angularVelocity = {0.0f, vx, 0.0f};
    state.inverseInertiaLocal = {0.25f + x, 0.5f + x, 0.75f + x};
    state.bodyType = (entityId % 2u == 0u) ? RigidBodyType::Kinematic : RigidBodyType::Dynamic;
    state.inverseMass = 1.0f;
    state.colliderShape = shape;
    state.colliderParams = {0.25f + x, 0.5f, 0.75f, 1.0f};
    state.kinematicTargetPosition = {x + 10.0f, 0.5f, -0.5f};
    state.kinematicTargetRotation = {0.0f, 0.0f, 0.3826834f, 0.9238795f};
    state.kinematicTargetEnabled = state.bodyType == RigidBodyType::Kinematic;
    return state;
}

bool verifySnapshotMatchesSoA(const PhysicsWorld& world)
{
    const auto& snapshot = world.rigidBodySnapshot();
    const auto& soa      = world.rigidBodySoA();
    if (snapshot.size() != soa.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < snapshot.size(); ++i)
    {
        const RigidBodyState& state = snapshot[i];
        if (state.entityId != soa.entityIds[i])
        {
            return false;
        }
        if (state.position.x != soa.positionsInvMass[i].x || state.position.y != soa.positionsInvMass[i].y ||
            state.position.z != soa.positionsInvMass[i].z || state.inverseMass != soa.positionsInvMass[i].w)
        {
            return false;
        }
        if (state.linearVelocity.x != soa.linearVelocities[i].x ||
            state.linearVelocity.y != soa.linearVelocities[i].y ||
            state.linearVelocity.z != soa.linearVelocities[i].z)
        {
            return false;
        }
        if (state.scale.x != soa.scales[i].x || state.scale.y != soa.scales[i].y ||
            state.scale.z != soa.scales[i].z)
        {
            return false;
        }
        if (state.angularVelocity.x != soa.angularVelocities[i].x ||
            state.angularVelocity.y != soa.angularVelocities[i].y ||
            state.angularVelocity.z != soa.angularVelocities[i].z)
        {
            return false;
        }
        if (state.inverseInertiaLocal.x != soa.inverseInertiaLocal[i].x ||
            state.inverseInertiaLocal.y != soa.inverseInertiaLocal[i].y ||
            state.inverseInertiaLocal.z != soa.inverseInertiaLocal[i].z)
        {
            return false;
        }
        if (static_cast<std::uint32_t>(state.bodyType) != soa.bodyTypes[i] ||
            static_cast<std::uint32_t>(state.colliderShape) != soa.colliderShapeTypes[i] ||
            state.colliderParams.x != soa.colliderParams[i].x ||
            state.kinematicTargetPosition.x != soa.kinematicTargetPositions[i].x ||
            state.kinematicTargetRotation.q.z != soa.kinematicTargetOrientations[i].z ||
            static_cast<std::uint32_t>(state.kinematicTargetEnabled) != soa.kinematicTargetFlags[i])
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    PhysicsWorld world;

    const EntityId e1 = 101;
    const EntityId e2 = 102;
    const EntityId e3 = 103;

    world.upsertRigidBody(makeRigidBody(e1, 1.0f, 2.0f, ColliderShapeType::Sphere));
    world.upsertRigidBody(makeRigidBody(e2, 2.0f, 3.0f, ColliderShapeType::Box));
    world.upsertRigidBody(makeRigidBody(e3, 3.0f, 4.0f, ColliderShapeType::Capsule));

    if (world.rigidBodyCount() != 3u || !verifySnapshotMatchesSoA(world))
    {
        std::cerr << "Initial SoA layout mismatch.\n";
        return 1;
    }

    if (!world.removeRigidBody(e2))
    {
        std::cerr << "Failed to remove middle body.\n";
        return 1;
    }
    if (world.rigidBodyCount() != 2u)
    {
        std::cerr << "Unexpected body count after removal.\n";
        return 1;
    }
    if (world.tryGetRigidBody(e2) != nullptr || world.tryGetRigidBody(e3) == nullptr)
    {
        std::cerr << "Entity-index remap failed after compaction.\n";
        return 1;
    }
    if (!verifySnapshotMatchesSoA(world))
    {
        std::cerr << "SoA/snapshot mismatch after compaction.\n";
        return 1;
    }

    world.upsertRigidBody(makeRigidBody(e3, 9.0f, 5.0f, ColliderShapeType::Capsule));
    const RigidBodyState* updated = world.tryGetRigidBody(e3);
    if (updated == nullptr || updated->position.x != 9.0f || updated->linearVelocity.x != 5.0f)
    {
        std::cerr << "Upsert update did not persist.\n";
        return 1;
    }
    if (!verifySnapshotMatchesSoA(world))
    {
        std::cerr << "SoA/snapshot mismatch after upsert update.\n";
        return 1;
    }

    std::cout << "Physics world SoA layout checks passed.\n";
    return 0;
}
