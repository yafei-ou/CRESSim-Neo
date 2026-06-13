#include "engine/world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId bodyA = world.createEntity();
    const common::EntityId bodyB = world.createEntity();

    physics::RigidBodyState rigidA{};
    rigidA.entityId = bodyA;
    rigidA.environmentIndex = 0u;
    rigidA.bodyType = physics::RigidBodyType::Dynamic;
    rigidA.inverseMass = 1.0f;
    world.physicsWorld().upsertRigidBody(rigidA);

    physics::RigidBodyState rigidB = rigidA;
    rigidB.entityId = bodyB;
    world.physicsWorld().upsertRigidBody(rigidB);

    physics::AuthoredRigidDistanceConstraintState constraint{};
    constraint.entityA = bodyA;
    constraint.entityB = bodyB;
    constraint.restDistance = 0.6f;

    const auto &authored = world.upsertRigidDistanceConstraint(constraint);
    const auto *roundTripped = world.tryGetRigidDistanceConstraint(authored.constraintId);
    if (roundTripped == nullptr || roundTripped->entityA != bodyA || roundTripped->entityB != bodyB)
    {
        CRESSIM_LOG_ERROR("Engine world rigid distance round-trip failed.\n");
        return 1;
    }

    if (!world.removeRigidDistanceConstraint(authored.constraintId) ||
        world.tryGetRigidDistanceConstraint(authored.constraintId) != nullptr)
    {
        CRESSIM_LOG_ERROR("Engine world rigid distance removal failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Engine rigid distance constraint authoring checks passed.\n");
    return 0;
}
