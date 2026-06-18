#include "common/logger.h"
#include "engine/world.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId bodyA = world.createEntity();
    const common::EntityId bodyB = world.createEntity();

    engine::RigidBodyComponent rigid{};
    rigid.bodyType = physics::RigidBodyType::Dynamic;
    world.setRigidBody(bodyA, rigid);
    world.setRigidBody(bodyB, rigid);

    physics::AuthoredRoutedCableConstraintState cable{};
    cable.routePoints = {
        {bodyA, {0.1f, 0.0f, 0.0f}},
        {bodyB, {-0.1f, 0.0f, 0.0f}},
    };
    cable.targetLength = 0.5f;

    physics::AuthoredRoutedCableConstraintState authored{};
    if (!world.upsertRoutedCableConstraint(cable, &authored))
    {
        CRESSIM_LOG_ERROR("Engine routed cable authoring failed.\n");
        return 1;
    }
    const auto *roundTripped = world.tryGetRoutedCableConstraint(authored.constraintId);
    if (roundTripped == nullptr || roundTripped->routePoints.size() != 2u ||
        roundTripped->targetLength != 0.5f)
    {
        CRESSIM_LOG_ERROR("Engine routed cable authoring round-trip failed.\n");
        return 1;
    }

    if (!world.removeRoutedCableConstraint(authored.constraintId) ||
        world.tryGetRoutedCableConstraint(authored.constraintId) != nullptr)
    {
        CRESSIM_LOG_ERROR("Engine routed cable removal failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Engine routed cable authoring checks passed.\n");
    return 0;
}
