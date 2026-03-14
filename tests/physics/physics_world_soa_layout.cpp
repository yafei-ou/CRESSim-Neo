#include "physics/physics_world.h"

#include <cstdint>
#include <iostream>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::ColliderState;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::RigidBodyState;
using cressim::neo::physics::RigidBodyType;

RigidBodyState makeRigidBody(EntityId entityId, float x, float vx)
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
    state.kinematicTargetPosition = {x + 10.0f, 0.5f, -0.5f};
    state.kinematicTargetRotation = {0.0f, 0.0f, 0.3826834f, 0.9238795f};
    state.kinematicTargetEnabled = state.bodyType == RigidBodyType::Kinematic;
    return state;
}

ColliderState makeCollider(EntityId entityId, float x, ColliderShapeType shape)
{
    ColliderState state{};
    state.entityId = entityId;
    state.shapeType = shape;
    state.shapeParams = {0.25f + x, 0.5f, 0.75f, 1.0f};
    state.localPosition = {0.1f * x, 0.0f, -0.1f * x};
    return state;
}

bool verifySnapshotMatchesSoA(const PhysicsWorld& world)
{
    const auto& bodySnapshot = world.rigidBodySnapshot();
    const auto& bodySoA = world.rigidBodySoA();
    if (bodySnapshot.size() != bodySoA.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < bodySnapshot.size(); ++i)
    {
        const RigidBodyState& state = bodySnapshot[i];
        if (state.rigidBodyId != bodySoA.rigidBodyIds[i] || state.entityId != bodySoA.entityIds[i] ||
            state.position.x != bodySoA.positionsInvMass[i].x ||
            state.position.y != bodySoA.positionsInvMass[i].y ||
            state.position.z != bodySoA.positionsInvMass[i].z ||
            state.inverseMass != bodySoA.positionsInvMass[i].w ||
            state.linearVelocity.x != bodySoA.linearVelocities[i].x ||
            state.scale.x != bodySoA.scales[i].x ||
            state.angularVelocity.y != bodySoA.angularVelocities[i].y ||
            state.inverseInertiaLocal.z != bodySoA.inverseInertiaLocal[i].z ||
            static_cast<std::uint32_t>(state.bodyType) != bodySoA.bodyTypes[i] ||
            state.kinematicTargetPosition.x != bodySoA.kinematicTargetPositions[i].x ||
            state.kinematicTargetRotation.q.z != bodySoA.kinematicTargetOrientations[i].z ||
            static_cast<std::uint32_t>(state.kinematicTargetEnabled) !=
                bodySoA.kinematicTargetFlags[i])
        {
            return false;
        }
    }

    const auto& colliderSnapshot = world.colliderSnapshot();
    const auto& colliderSoA = world.colliderSoA();
    const auto& bodyMapping = world.bodyColliderMapping();
    if (colliderSnapshot.size() != colliderSoA.size() ||
        bodyMapping.colliderCounts.size() != bodySnapshot.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < colliderSnapshot.size(); ++i)
    {
        const ColliderState& state = colliderSnapshot[i];
        if (state.colliderId != colliderSoA.colliderIds[i] ||
            state.entityId != colliderSoA.entityIds[i] ||
            state.ownerRigidBodyId != colliderSoA.ownerRigidBodyIds[i] ||
            static_cast<std::uint32_t>(state.shapeType) != colliderSoA.shapeTypes[i] ||
            state.shapeParams.x != colliderSoA.shapeParams[i].x ||
            state.localPosition.x != colliderSoA.localPositions[i].x)
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

    world.upsertRigidBody(makeRigidBody(e1, 1.0f, 2.0f));
    world.replaceColliders(e1, {makeCollider(e1, 1.0f, ColliderShapeType::Sphere)});
    world.upsertRigidBody(makeRigidBody(e2, 2.0f, 3.0f));
    world.replaceColliders(e2, {makeCollider(e2, 2.0f, ColliderShapeType::Box)});
    world.upsertRigidBody(makeRigidBody(e3, 3.0f, 4.0f));
    world.replaceColliders(e3, {makeCollider(e3, 3.0f, ColliderShapeType::Capsule)});

    if (world.rigidBodyCount() != 3u || world.colliderCount() != 3u ||
        !verifySnapshotMatchesSoA(world))
    {
        std::cerr << "Initial SoA layout mismatch.\n";
        return 1;
    }

    if (!world.removeRigidBody(e2))
    {
        std::cerr << "Failed to remove middle body.\n";
        return 1;
    }
    if (world.rigidBodyCount() != 2u || world.colliderCount() != 2u)
    {
        std::cerr << "Unexpected body/collider count after removal.\n";
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

    world.upsertRigidBody(makeRigidBody(e3, 9.0f, 5.0f));
    world.replaceColliders(e3, {makeCollider(e3, 9.0f, ColliderShapeType::Capsule),
                                makeCollider(e3, 9.5f, ColliderShapeType::Sphere)});
    const RigidBodyState* updated = world.tryGetRigidBody(e3);
    if (updated == nullptr || updated->position.x != 9.0f || updated->linearVelocity.x != 5.0f)
    {
        std::cerr << "Upsert update did not persist.\n";
        return 1;
    }
    if (world.colliderCount() != 3u || !verifySnapshotMatchesSoA(world))
    {
        std::cerr << "SoA/snapshot mismatch after upsert update.\n";
        return 1;
    }

    std::cout << "Physics world SoA layout checks passed.\n";
    return 0;
}
