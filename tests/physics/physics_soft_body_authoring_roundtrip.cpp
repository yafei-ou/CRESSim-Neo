#include "common/logger.h"
#include "common/scene_primitives.h"
#include "engine/components.h"
#include "engine/world.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    common::SceneLayoutDesc sceneLayout{};
    sceneLayout.envCount = 4u;
    world.setSceneLayout(sceneLayout);
    const common::EntityId entity = world.createEntity(3u);

    engine::TransformComponent transform{};
    transform.worldTransform.position = {2.0f, 4.0f, 6.0f};
    world.setTransform(entity, transform);

    engine::SoftBodyComponent softBody{};
    softBody.size                 = {1.5f, 2.0f, 2.5f};
    softBody.particleSpacing      = 0.4f;
    softBody.particleMass         = 0.75f;
    softBody.particleRadius       = 0.18f;
    softBody.edgeCompliance       = 0.03f;
    softBody.volumeCompliance     = 0.04f;
    softBody.simulated            = true;
    softBody.selfCollisionEnabled = true;
    softBody.collisionLayer       = 0x4u;
    softBody.collisionMask        = 0x12u;
    world.setSoftBody(entity, softBody);

    const std::optional<engine::SoftBodyComponent> roundTripped = world.tryGetSoftBody(entity);
    if (!roundTripped.has_value())
    {
        CRESSIM_LOG_ERROR("Soft body round-trip lookup returned no component.");
        return 1;
    }

    const auto *storedState = world.physicsWorld().tryGetSoftBody(entity);
    if (storedState == nullptr)
    {
        CRESSIM_LOG_ERROR("PhysicsWorld is missing the authored soft body.");
        return 1;
    }

    if (!roundTripped->selfCollisionEnabled || roundTripped->collisionLayer != 0x4u ||
        roundTripped->collisionMask != 0x12u || !storedState->selfCollisionEnabled ||
        storedState->collisionLayer != 0x4u || storedState->collisionMask != 0x12u ||
        storedState->environmentIndex != 3u)
    {
        CRESSIM_LOG_ERROR("Soft body collision authoring fields did not round-trip.");
        return 1;
    }

    CRESSIM_LOG_INFO("Soft body authoring round-trip checks passed.");
    return 0;
}
