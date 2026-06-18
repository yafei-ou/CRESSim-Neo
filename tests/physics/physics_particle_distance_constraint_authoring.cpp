#include "physics/physics_world.h"
#include "common/logger.h"

#include <cmath>

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::RigidBodyState needle{};
    needle.entityId = 9001u;
    needle.environmentIndex = 3u;
    needle.bodyType = physics::RigidBodyType::Kinematic;
    needle.proxyParticleRadius = 0.05f;
    needle.proxyParticleLocalPositions = {
        {-0.2f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.2f, 0.0f, 0.0f},
    };
    world.upsertRigidBody(needle);

    physics::StrandState strand{};
    strand.entityId = 9002u;
    strand.environmentIndex = 3u;
    strand.stretchShearCompliance = 0.02f;
    strand.bendCompliance = 0.01f;
    strand.restPositions = {
        {0.2f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
        {0.8f, 0.0f, 0.0f},
    };
    if (!world.upsertStrand(strand))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for particle distance constraint test.\n");
        return 1;
    }

    physics::AuthoredParticleDistanceConstraintState attachment{};
    attachment.particleA.entityId = needle.entityId;
    attachment.particleA.type = physics::AuthoredParticleReferenceType::RigidProxyParticle;
    attachment.particleA.localParticleIndex = 2u;
    attachment.particleB.entityId = strand.entityId;
    attachment.particleB.type = physics::AuthoredParticleReferenceType::StrandParticle;
    attachment.particleB.localParticleIndex = 0u;
    attachment.restLength = 0.0f;
    attachment.compliance = 0.005f;

    const auto &authored = world.upsertParticleDistanceConstraint(attachment);
    if (authored.constraintId == physics::kInvalidParticleConstraintId)
    {
        CRESSIM_LOG_ERROR("Particle distance constraint id was not assigned.\n");
        return 1;
    }

    if (world.particleDistanceConstraintSnapshot().size() != 1u)
    {
        CRESSIM_LOG_ERROR("Particle distance constraint snapshot was not updated.\n");
        return 1;
    }

    const physics::RigidBodyState *authoredNeedle = world.tryGetRigidBody(needle.entityId);
    const physics::StrandState *authoredStrand = world.tryGetStrand(strand.entityId);
    if (authoredNeedle == nullptr || authoredStrand == nullptr)
    {
        CRESSIM_LOG_ERROR("Failed to retrieve authored rigid/strand state.\n");
        return 1;
    }

    const auto &constraints = world.distanceConstraints();
    if (constraints.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Resolved distance constraints did not include authored attachment.\n");
        return 1;
    }

    const physics::DeformableDistanceConstraint &attachmentConstraint = constraints.back();
    const std::uint32_t expectedNeedleParticle = authoredNeedle->proxyParticleOffset + 2u;
    const std::uint32_t expectedStrandParticle = authoredStrand->particleOffset + 0u;
    if (attachmentConstraint.particleA != expectedNeedleParticle ||
        attachmentConstraint.particleB != expectedStrandParticle ||
        std::abs(attachmentConstraint.restLength) > 1.0e-6f ||
        std::abs(attachmentConstraint.compliance - 0.005f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Resolved authored attachment constraint is incorrect.\n");
        return 1;
    }

    physics::AuthoredParticleDistanceConstraintState updated = authored;
    updated.compliance = 0.01f;
    updated.enabled = false;
    world.upsertParticleDistanceConstraint(updated);

    if (world.distanceConstraints().size() != 0u)
    {
        CRESSIM_LOG_ERROR("Disabled authored particle constraint should not resolve.\n");
        return 1;
    }

    updated.enabled = true;
    world.upsertParticleDistanceConstraint(updated);
    if (world.distanceConstraints().size() != 1u ||
        std::abs(world.distanceConstraints().back().compliance - 0.01f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Updated authored particle constraint did not rebuild correctly.\n");
        return 1;
    }

    if (!world.removeParticleDistanceConstraint(authored.constraintId) ||
        !world.particleDistanceConstraintSnapshot().empty() ||
        !world.distanceConstraints().empty())
    {
        CRESSIM_LOG_ERROR("Removing authored particle distance constraint failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Particle distance constraint authoring checks passed.\n");
    return 0;
}
