#include "physics/physics_world.h"
#include "common/logger.h"

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::RigidBodyState;
using cressim::neo::physics::RigidBodyType;

RigidBodyState makeBody(EntityId entityId, float x)
{
    RigidBodyState state{};
    state.entityId = entityId;
    state.environmentIndex = 0u;
    state.position = {x, 0.0f, 0.0f};
    state.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.scale = {1.0f, 1.0f, 1.0f};
    state.inverseMass = 1.0f;
    state.inverseInertiaLocal = {1.0f, 1.0f, 1.0f};
    state.bodyType = RigidBodyType::Dynamic;
    return state;
}

} // namespace

int main()
{
    PhysicsWorld world;

    world.upsertRigidBody(makeBody(1001u, 0.0f));
    const auto &initialSuppression = world.jointCollisionSuppression();
    if (initialSuppression.neighborOffsets.size() != 2u ||
        initialSuppression.neighborOffsets[0] != 0u ||
        initialSuppression.neighborOffsets[1] != 0u)
    {
        CRESSIM_LOG_ERROR("Initial joint suppression layout did not match the rigid body count.");
        return 1;
    }

    world.upsertRigidBody(makeBody(1002u, 1.0f));
    const auto &grownSuppression = world.jointCollisionSuppression();
    if (grownSuppression.neighborOffsets.size() != 3u ||
        grownSuppression.neighborOffsets[0] != 0u ||
        grownSuppression.neighborOffsets[1] != 0u ||
        grownSuppression.neighborOffsets[2] != 0u)
    {
        CRESSIM_LOG_ERROR(
            "Joint suppression offsets were not rebuilt after rigid body insertion.");
        return 1;
    }

    return 0;
}
