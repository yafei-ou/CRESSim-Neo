#include "engine/world.h"
#include "common/logger.h"

#include <cmath>

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId needleEntity = world.createEntity();
    const common::EntityId strandEntity = world.createEntity();

    engine::RigidBodyComponent needle{};
    needle.bodyType = physics::RigidBodyType::Kinematic;
    needle.proxyParticleRadius = 0.05f;
    needle.proxyParticleLocalPositions = {
        {-0.2f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.2f, 0.0f, 0.0f},
    };
    world.setRigidBody(needleEntity, needle);

    engine::StrandComponent strand{};
    strand.restPositions = {
        {0.2f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
        {0.8f, 0.0f, 0.0f},
    };
    if (!world.setStrand(strandEntity, strand))
    {
        CRESSIM_LOG_ERROR("setStrand failed in particle distance constraint test.\n");
        return 1;
    }

    physics::AuthoredParticleDistanceConstraintState attachment{};
    attachment.particleA.entityId = needleEntity;
    attachment.particleA.type = physics::AuthoredParticleReferenceType::RigidProxyParticle;
    attachment.particleA.localParticleIndex = 2u;
    attachment.particleB.entityId = strandEntity;
    attachment.particleB.type = physics::AuthoredParticleReferenceType::StrandParticle;
    attachment.particleB.localParticleIndex = 0u;
    attachment.restLength = 0.0f;
    attachment.compliance = 0.015f;

    const auto &authored = world.upsertParticleDistanceConstraint(attachment);
    const auto *roundTripped = world.tryGetParticleDistanceConstraint(authored.constraintId);
    if (roundTripped == nullptr || std::abs(roundTripped->compliance - 0.015f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("World particle distance constraint authoring did not round-trip.\n");
        return 1;
    }

    const auto &constraints = world.physicsWorld().distanceConstraints();
    if (constraints.size() != 3u ||
        std::abs(constraints.back().compliance - 0.015f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Physics world did not resolve authored attachment constraint.\n");
        return 1;
    }

    if (!world.removeParticleDistanceConstraint(authored.constraintId) ||
        world.tryGetParticleDistanceConstraint(authored.constraintId) != nullptr)
    {
        CRESSIM_LOG_ERROR("World failed to remove authored particle distance constraint.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Particle distance constraint world checks passed.\n");
    return 0;
}
