#include "engine/components.h"
#include "engine/world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId entity = world.createEntity();

    engine::RigidBodyComponent rigidBody{};
    rigidBody.bodyType = physics::RigidBodyType::Kinematic;
    rigidBody.inverseMass = 0.5f;
    rigidBody.proxyParticleRadius = 0.08f;
    rigidBody.proxyParticleLocalPositions = {
        {-0.4f, 0.0f, 0.0f},
        {0.0f, 0.2f, 0.0f},
        {0.4f, 0.0f, 0.0f},
    };
    rigidBody.proxyCollisionLayer = 0x10u;
    rigidBody.proxyCollisionMask = 0x24u;
    rigidBody.suturingEnabled = true;
    rigidBody.needleTipProxyIndex = 2u;
    rigidBody.kinematicTargetEnabled = true;
    rigidBody.kinematicTargetPosition = {1.0f, 2.0f, 3.0f};

    world.setRigidBody(entity, rigidBody);

    const physics::RigidBodyState *authored = world.physicsWorld().tryGetRigidBody(entity);
    if (authored == nullptr || !authored->suturingEnabled ||
        authored->needleTipProxyIndex != rigidBody.needleTipProxyIndex ||
        authored->proxyParticleLocalPositions.size() != rigidBody.proxyParticleLocalPositions.size())
    {
        CRESSIM_LOG_ERROR("Physics world did not preserve rigid proxy suturing metadata.\n");
        return 1;
    }

    const auto component = world.tryGetRigidBody(entity);
    if (!component.has_value() || !component->suturingEnabled ||
        component->needleTipProxyIndex != rigidBody.needleTipProxyIndex ||
        component->proxyParticleLocalPositions.size() != rigidBody.proxyParticleLocalPositions.size())
    {
        CRESSIM_LOG_ERROR("World::tryGetRigidBody did not round-trip rigid proxy suturing data.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Rigid body proxy suturing component checks passed.\n");
    return 0;
}
