#include "engine/components.h"
#include "engine/world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId entity = world.createEntity();

    engine::StrandComponent strand{};
    strand.restPositions = {
        {-0.5f, 1.0f, 0.0f},
        {0.0f, 0.9f, 0.0f},
        {0.5f, 0.8f, 0.0f},
    };
    strand.staticParticleIndices = {0u};
    strand.particleMass          = 0.4f;
    strand.particleRadius        = 0.06f;
    strand.distanceCompliance    = 0.02f;
    strand.bendCompliance        = 0.01f;
    strand.selfCollisionEnabled  = true;
    strand.collisionLayer        = 0x10u;
    strand.collisionMask         = 0x22u;
    strand.suturingEnabled       = true;
    strand.needleTipParticleIndex = 2u;
    strand.needleTipKinematic    = true;
    strand.pathNodeSpacing       = 0.18f;

    if (!world.setStrand(entity, strand))
    {
        CRESSIM_LOG_ERROR("setStrand failed.\n");
        return 1;
    }

    const physics::PhysicsWorld &physicsWorld = world.physicsWorld();
    const physics::StrandState *authored      = physicsWorld.tryGetStrand(entity);
    if (authored == nullptr || physicsWorld.strandCount() != 1u ||
        authored->environmentIndex != 0u || authored->restPositions.size() != 3u)
    {
        CRESSIM_LOG_ERROR("Physics world did not preserve authored strand state.\n");
        return 1;
    }

    if (!authored->suturingEnabled || authored->needleTipParticleIndex != 2u ||
        !authored->needleTipKinematic || std::abs(authored->pathNodeSpacing - 0.18f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Physics world did not preserve strand suturing fields.\n");
        return 1;
    }

    const auto component = world.tryGetStrand(entity);
    if (!component.has_value() || component->restPositions.size() != 3u ||
        component->distanceCompliance != strand.distanceCompliance ||
        component->bendCompliance != strand.bendCompliance ||
        component->suturingEnabled != strand.suturingEnabled ||
        component->needleTipParticleIndex != strand.needleTipParticleIndex ||
        component->needleTipKinematic != strand.needleTipKinematic ||
        std::abs(component->pathNodeSpacing - strand.pathNodeSpacing) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("World::tryGetStrand did not round-trip strand data.\n");
        return 1;
    }

    if (!world.setEntityEnvironment(entity, 0u))
    {
        CRESSIM_LOG_ERROR("setEntityEnvironment to current env unexpectedly failed.\n");
        return 1;
    }

    if (!world.removeStrand(entity) || physicsWorld.tryGetStrand(entity) != nullptr ||
        physicsWorld.strandCount() != 0u)
    {
        CRESSIM_LOG_ERROR("removeStrand failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Strand component authoring checks passed.\n");
    return 0;
}
