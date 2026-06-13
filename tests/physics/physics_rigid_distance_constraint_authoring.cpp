#include "physics/physics_world.h"
#include "common/logger.h"

#include <cmath>

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::RigidBodyState bodyA{};
    bodyA.entityId = 8201u;
    bodyA.environmentIndex = 3u;
    bodyA.bodyType = physics::RigidBodyType::Dynamic;
    bodyA.inverseMass = 1.0f;
    world.upsertRigidBody(bodyA);

    physics::RigidBodyState bodyB = bodyA;
    bodyB.entityId = 8202u;
    bodyB.position = {0.0f, -1.0f, 0.0f};
    world.upsertRigidBody(bodyB);

    physics::AuthoredRigidDistanceConstraintState constraint{};
    constraint.entityA = bodyA.entityId;
    constraint.entityB = bodyB.entityId;
    constraint.localAnchorA = {0.0f, -0.1f, 0.0f};
    constraint.localAnchorB = {0.0f, 0.1f, 0.0f};
    constraint.restDistance = 0.8f;
    constraint.compliance = 0.01f;

    const auto &authored = world.upsertRigidDistanceConstraint(constraint);
    if (authored.constraintId == physics::kInvalidRigidDistanceConstraintId)
    {
        CRESSIM_LOG_ERROR("Rigid distance constraint id was not assigned.\n");
        return 1;
    }

    const auto &resolved = world.rigidDistanceConstraints();
    if (resolved.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Rigid distance constraint did not resolve.\n");
        return 1;
    }

    if (resolved[0].rigidBodyIndexA != 0u || resolved[0].rigidBodyIndexB != 1u ||
        std::abs(resolved[0].restDistance - 0.8f) > 1.0e-6f ||
        resolved[0].localAnchorA.y != -0.1f || resolved[0].localAnchorB.y != 0.1f)
    {
        CRESSIM_LOG_ERROR("Resolved rigid distance constraint payload is incorrect.\n");
        return 1;
    }

    const std::uint64_t topologyRevision = world.rigidDistanceConstraintTopologyRevision();
    const std::uint64_t payloadRevision = world.rigidDistanceConstraintRevision();
    physics::AuthoredRigidDistanceConstraintState updated = authored;
    updated.restDistance = 0.5f;
    updated.enabled = false;
    world.upsertRigidDistanceConstraint(updated);
    if (world.rigidDistanceConstraintTopologyRevision() != topologyRevision ||
        world.rigidDistanceConstraintRevision() == payloadRevision ||
        !world.rigidDistanceConstraints().empty())
    {
        CRESSIM_LOG_ERROR("Rigid distance runtime payload update behaved incorrectly.\n");
        return 1;
    }

    updated.enabled = true;
    updated.localAnchorB = {0.0f, 0.2f, 0.0f};
    world.upsertRigidDistanceConstraint(updated);
    if (world.rigidDistanceConstraintTopologyRevision() == topologyRevision ||
        world.rigidDistanceConstraints().size() != 1u ||
        world.rigidDistanceConstraints()[0].localAnchorB.y != 0.2f)
    {
        CRESSIM_LOG_ERROR("Rigid distance topology update did not rebuild.\n");
        return 1;
    }

    physics::RigidBodyState bodyC = bodyA;
    bodyC.entityId = 8203u;
    bodyC.environmentIndex = 5u;
    world.upsertRigidBody(bodyC);
    physics::AuthoredRigidDistanceConstraintState invalid{};
    invalid.entityA = bodyA.entityId;
    invalid.entityB = bodyC.entityId;
    world.upsertRigidDistanceConstraint(invalid);
    if (world.rigidDistanceConstraints().size() != 1u)
    {
        CRESSIM_LOG_ERROR("Invalid cross-environment rigid distance should not resolve.\n");
        return 1;
    }

    const auto authoredEntries = world.rigidDistanceConstraintSnapshot();
    for (const auto &entry : authoredEntries)
    {
        if (!world.removeRigidDistanceConstraint(entry.constraintId))
        {
            CRESSIM_LOG_ERROR("Removing rigid distance constraint entry failed.\n");
            return 1;
        }
    }

    if (!world.rigidDistanceConstraintSnapshot().empty() || !world.rigidDistanceConstraints().empty())
    {
        CRESSIM_LOG_ERROR("Removing rigid distance constraint failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Rigid distance constraint authoring checks passed.\n");
    return 0;
}
