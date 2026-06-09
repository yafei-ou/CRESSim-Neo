#include "engine/world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId needleEntity = world.createEntity();
    const common::EntityId strandEntity = world.createEntity();

    engine::RigidBodyComponent needle{};
    needle.bodyType = physics::RigidBodyType::Kinematic;
    needle.proxyParticleRadius = 0.04f;
    needle.proxyParticleLocalPositions = {
        {-0.1f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.1f, 0.0f, 0.0f},
    };
    world.setRigidBody(needleEntity, needle);

    engine::StrandComponent strand{};
    strand.restPositions = {
        {0.1f, 0.0f, 0.0f},
        {0.24f, 0.0f, 0.0f},
        {0.38f, 0.0f, 0.0f},
    };
    if (!world.setStrand(strandEntity, strand))
    {
        CRESSIM_LOG_ERROR("setStrand failed in suturing sequence test.\n");
        return 1;
    }

    physics::AuthoredSuturingSequenceState sequence{};
    sequence.entries = {
        {needleEntity, physics::AuthoredParticleReferenceType::RigidProxyParticle, 0u},
        {needleEntity, physics::AuthoredParticleReferenceType::RigidProxyParticle, 1u},
        {needleEntity, physics::AuthoredParticleReferenceType::RigidProxyParticle, 2u},
        {strandEntity, physics::AuthoredParticleReferenceType::StrandParticle, 0u},
        {strandEntity, physics::AuthoredParticleReferenceType::StrandParticle, 1u},
        {strandEntity, physics::AuthoredParticleReferenceType::StrandParticle, 2u},
    };
    sequence.pathNodeSpacing = 0.19f;

    const auto &authored = world.upsertSuturingSequence(sequence);
    const auto *roundTripped = world.tryGetSuturingSequence(authored.sequenceId);
    if (roundTripped == nullptr || roundTripped->entries.size() != 6u ||
        roundTripped->pathNodeSpacing != 0.19f)
    {
        CRESSIM_LOG_ERROR("World suturing sequence authoring did not round-trip.\n");
        return 1;
    }

    if (!world.removeSuturingSequence(authored.sequenceId) ||
        world.tryGetSuturingSequence(authored.sequenceId) != nullptr)
    {
        CRESSIM_LOG_ERROR("World failed to remove authored suturing sequence.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Suturing sequence world checks passed.\n");
    return 0;
}
