#include "physics/physics_world.h"

#include <cmath>
#include <iostream>

namespace
{

using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::RigidBodyState;
using cressim::neo::physics::RigidBodyType;

RigidBodyState makeBody(cressim::neo::common::EntityId id, RigidBodyType type)
{
    RigidBodyState state{};
    state.entityId = id;
    state.bodyType = type;
    state.position = {1.0f, 2.0f, 3.0f};
    state.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.scale = {1.0f, 1.0f, 1.0f};
    state.linearVelocity = {0.5f, 0.0f, 0.0f};
    state.angularVelocity = {0.0f, 1.0f, 0.0f};
    state.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    state.inverseMass = 1.0f;
    state.colliderShape = ColliderShapeType::Box;
    state.colliderParams = {0.5f, 0.5f, 0.5f, 0.0f};
    state.kinematicTargetPosition = {4.0f, 5.0f, 6.0f};
    state.kinematicTargetRotation = {0.0f, 0.0f, 0.3826834f, 0.9238795f};
    state.kinematicTargetEnabled = type == RigidBodyType::Kinematic;
    return state;
}

} // namespace

int main()
{
    using namespace cressim::neo;

    PhysicsWorld world;

    RigidBodyState dynamicBody = makeBody(4001u, RigidBodyType::Dynamic);
    dynamicBody.inverseMass = 0.0f;
    world.upsertRigidBody(dynamicBody);
    const RigidBodyState* normalizedDynamic = world.tryGetRigidBody(4001u);
    if (normalizedDynamic == nullptr || normalizedDynamic->inverseMass <= 0.0f)
    {
        std::cerr << "Dynamic body inverse-mass normalization failed.\n";
        return 1;
    }

    RigidBodyState staticBody = makeBody(4002u, RigidBodyType::Static);
    staticBody.inverseMass = 5.0f;
    staticBody.position = {0.0f, -1.0f, 0.0f};
    world.upsertRigidBody(staticBody);
    if (!world.staticBroadPhaseDirty())
    {
        std::cerr << "Static broad-phase was not marked dirty for a static body insert.\n";
        return 1;
    }
    world.clearStaticBroadPhaseDirty();

    RigidBodyState kinematicBody = makeBody(4003u, RigidBodyType::Kinematic);
    world.upsertRigidBody(kinematicBody);
    const RigidBodyState* persistedKinematic = world.tryGetRigidBody(4003u);
    if (persistedKinematic == nullptr || !persistedKinematic->kinematicTargetEnabled ||
        std::fabs(persistedKinematic->kinematicTargetPosition.x -
                  kinematicBody.kinematicTargetPosition.x) > 1e-5f)
    {
        std::cerr << "Kinematic target state was not preserved.\n";
        return 1;
    }

    staticBody.position.x = 2.0f;
    world.upsertRigidBody(staticBody);
    if (!world.staticBroadPhaseDirty())
    {
        std::cerr << "Static broad-phase was not marked dirty for a static shape update.\n";
        return 1;
    }

    const auto& soa = world.rigidBodySoA();
    if (soa.bodyTypes.size() != world.rigidBodyCount() ||
        soa.kinematicTargetFlags.size() != world.rigidBodyCount())
    {
        std::cerr << "SoA body type or kinematic target arrays are out of sync.\n";
        return 1;
    }

    std::cout << "Physics body type checks passed.\n";
    return 0;
}
