#include "physics/physics_world.h"
#include "common/logger.h"

#include <cmath>

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::RigidBodyState bodyA{};
    bodyA.entityId = 8101u;
    bodyA.environmentIndex = 2u;
    bodyA.bodyType = physics::RigidBodyType::Dynamic;
    bodyA.inverseMass = 1.0f;
    world.upsertRigidBody(bodyA);

    physics::RigidBodyState bodyB = bodyA;
    bodyB.entityId = 8102u;
    bodyB.position = {1.0f, 0.0f, 0.0f};
    world.upsertRigidBody(bodyB);

    physics::RigidBodyState bodyC = bodyA;
    bodyC.entityId = 8103u;
    bodyC.position = {2.0f, 0.0f, 0.0f};
    world.upsertRigidBody(bodyC);

    physics::AuthoredRoutedCableConstraintState cable{};
    cable.routePoints = {
        {bodyA.entityId, {0.1f, 0.0f, 0.0f}},
        {bodyB.entityId, {0.1f, 0.1f, 0.0f}},
        {bodyC.entityId, {0.1f, 0.0f, 0.0f}},
    };
    cable.targetLength = 1.5f;
    cable.compliance = 0.01f;

    physics::AuthoredRoutedCableConstraintState authored{};
    if (!world.upsertRoutedCableConstraint(cable, &authored) ||
        authored.constraintId == physics::kInvalidRoutedCableConstraintId)
    {
        CRESSIM_LOG_ERROR("Routed cable constraint id was not assigned.\n");
        return 1;
    }

    const auto &resolvedConstraints = world.routedCableConstraints();
    const auto &resolvedRoutePoints = world.routedCableRoutePoints();
    if (resolvedConstraints.size() != 1u || resolvedRoutePoints.size() != 3u)
    {
        CRESSIM_LOG_ERROR("Resolved routed cable topology counts are incorrect.\n");
        return 1;
    }

    if (resolvedConstraints[0].routePointCount != 3u ||
        std::abs(resolvedConstraints[0].targetLength - 1.5f) > 1.0e-6f ||
        resolvedRoutePoints[1].rigidBodyIndex != 1u ||
        resolvedRoutePoints[1].localGuideOffset.x != 0.1f ||
        resolvedRoutePoints[1].localGuideOffset.y != 0.1f)
    {
        CRESSIM_LOG_ERROR("Resolved routed cable topology payload is incorrect.\n");
        return 1;
    }

    const std::uint64_t definitionRevision = world.routedCableDefinitionRevision();
    const std::uint64_t resolvedRevision = world.routedCableResolvedRevision();
    physics::AuthoredRoutedCableConstraintState updated = authored;
    updated.targetLength = 1.0f;
    updated.enabled = false;
    if (!world.upsertRoutedCableConstraint(updated) ||
        world.routedCableDefinitionRevision() == definitionRevision ||
        !world.routedCableConstraints().empty())
    {
        CRESSIM_LOG_ERROR("Runtime routed cable payload update behaved incorrectly.\n");
        return 1;
    }
    if (world.routedCableResolvedRevision() == resolvedRevision)
    {
        CRESSIM_LOG_ERROR("Resolved routed cable revision did not rebuild.\n");
        return 1;
    }

    updated.enabled = true;
    updated.routePoints.push_back({bodyA.entityId, {0.0f, 0.2f, 0.0f}});
    if (!world.upsertRoutedCableConstraint(updated) ||
        world.routedCableRoutePoints().size() != 4u)
    {
        CRESSIM_LOG_ERROR("Routed cable update did not rebuild.\n");
        return 1;
    }

    bodyB.environmentIndex = 7u;
    world.upsertRigidBody(bodyB);
    if (!world.routedCableConstraints().empty() || !world.routedCableRoutePoints().empty())
    {
        CRESSIM_LOG_ERROR("Routed cable should be dropped after body environment migration.\n");
        return 1;
    }

    bodyB.environmentIndex = bodyA.environmentIndex;
    world.upsertRigidBody(bodyB);
    if (world.routedCableConstraints().size() != 1u || world.routedCableRoutePoints().size() != 4u)
    {
        CRESSIM_LOG_ERROR("Routed cable did not rebuild after body returned to the environment.\n");
        return 1;
    }

    physics::RigidBodyState bodyD = bodyA;
    bodyD.entityId = 8104u;
    bodyD.environmentIndex = 5u;
    world.upsertRigidBody(bodyD);
    physics::AuthoredRoutedCableConstraintState invalid{};
    invalid.routePoints = {
        {bodyA.entityId, {0.0f, 0.0f, 0.0f}},
        {bodyD.entityId, {0.0f, 0.0f, 0.0f}},
    };
    if (world.upsertRoutedCableConstraint(invalid) ||
        world.routedCableConstraintSnapshot().size() != 1u ||
        world.routedCableConstraints().size() != 1u)
    {
        CRESSIM_LOG_ERROR("Invalid cross-environment routed cable should be rejected.\n");
        return 1;
    }

    const auto authoredIds = world.routedCableConstraintSnapshot();
    for (const auto &entry : authoredIds)
    {
        if (!world.removeRoutedCableConstraint(entry.constraintId))
        {
            CRESSIM_LOG_ERROR("Removing routed cable constraint entry failed.\n");
            return 1;
        }
    }

    if (!world.routedCableConstraintSnapshot().empty() || !world.routedCableConstraints().empty())
    {
        CRESSIM_LOG_ERROR("Removing routed cable constraint failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Routed cable authoring checks passed.\n");
    return 0;
}
