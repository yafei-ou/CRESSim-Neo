#include "common/logger.h"
#include "physics/physics_world.h"

#include <cmath>

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::RigidBodyState disk{};
    disk.entityId         = 9101u;
    disk.environmentIndex = 2u;
    disk.bodyType         = physics::RigidBodyType::Dynamic;
    disk.inverseMass      = 1.0f;
    world.upsertRigidBody(disk);

    physics::StrandState backbone{};
    backbone.entityId         = 9102u;
    backbone.environmentIndex = 2u;
    backbone.restPositions = {
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.5f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    backbone.staticParticleIndices = {0u};
    if (!world.upsertStrand(backbone))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for rigid-particle attachment test.\n");
        return 1;
    }

    physics::AuthoredRigidParticleAttachmentConstraintState attachment{};
    attachment.particle.entityId          = backbone.entityId;
    attachment.particle.type              = physics::AuthoredParticleReferenceType::StrandParticle;
    attachment.particle.localParticleIndex = 1u;
    attachment.rigidBodyEntityId          = disk.entityId;
    attachment.localAnchor                = {0.0f, 0.15f, 0.0f};
    attachment.compliance                 = 0.02f;

    const auto &authored = world.upsertRigidParticleAttachmentConstraint(attachment);
    if (authored.constraintId == physics::kInvalidRigidParticleAttachmentConstraintId)
    {
        CRESSIM_LOG_ERROR("Rigid-particle attachment id was not assigned.\n");
        return 1;
    }

    const auto &resolved = world.rigidParticleAttachments();
    if (resolved.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Rigid-particle attachment did not resolve.\n");
        return 1;
    }

    const physics::StrandState *resolvedBackbone = world.tryGetStrand(backbone.entityId);
    if (resolvedBackbone == nullptr ||
        resolved[0].particleIndex != resolvedBackbone->particleOffset + 1u ||
        resolved[0].rigidBodyIndex != 0u ||
        std::abs(resolved[0].compliance - 0.02f) > 1.0e-6f ||
        std::abs(resolved[0].localAnchor.y - 0.15f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Resolved rigid-particle attachment payload is incorrect.\n");
        return 1;
    }

    const std::uint64_t topologyRevision = world.rigidParticleAttachmentTopologyRevision();
    const std::uint64_t payloadRevision  = world.rigidParticleAttachmentRevision();
    physics::AuthoredRigidParticleAttachmentConstraintState updated = authored;
    updated.compliance = 0.05f;
    updated.enabled    = false;
    world.upsertRigidParticleAttachmentConstraint(updated);
    if (world.rigidParticleAttachmentTopologyRevision() != topologyRevision ||
        world.rigidParticleAttachmentRevision() == payloadRevision ||
        !world.rigidParticleAttachments().empty())
    {
        CRESSIM_LOG_ERROR("Rigid-particle attachment runtime payload update behaved incorrectly.\n");
        return 1;
    }

    updated.enabled                = true;
    updated.rigidBodyEntityId      = 0u;
    world.upsertRigidParticleAttachmentConstraint(updated);
    if (world.rigidParticleAttachmentTopologyRevision() == topologyRevision ||
        !world.rigidParticleAttachments().empty())
    {
        CRESSIM_LOG_ERROR("Invalid rigid-particle attachment topology update did not rebuild.\n");
        return 1;
    }

    if (!world.removeRigidParticleAttachmentConstraint(authored.constraintId) ||
        !world.rigidParticleAttachmentConstraintSnapshot().empty() ||
        !world.rigidParticleAttachments().empty())
    {
        CRESSIM_LOG_ERROR("Removing rigid-particle attachment failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Rigid-particle attachment authoring checks passed.\n");
    return 0;
}
