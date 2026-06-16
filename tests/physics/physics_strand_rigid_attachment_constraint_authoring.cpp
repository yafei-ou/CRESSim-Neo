#include "common/logger.h"
#include "physics/physics_world.h"

#include <cmath>

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::RigidBodyState disk{};
    disk.entityId         = 9201u;
    disk.environmentIndex = 3u;
    disk.bodyType         = physics::RigidBodyType::Dynamic;
    disk.inverseMass      = 1.0f;
    world.upsertRigidBody(disk);

    physics::StrandState backbone{};
    backbone.entityId         = 9202u;
    backbone.environmentIndex = 3u;
    backbone.restPositions = {
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.5f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    backbone.staticParticleIndices = {0u};
    if (!world.upsertStrand(backbone))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for strand-rigid attachment test.\n");
        return 1;
    }

    physics::AuthoredStrandRigidAttachmentConstraintState attachment{};
    attachment.strandEntityId         = backbone.entityId;
    attachment.localSegmentIndex      = 1u;
    attachment.segmentT               = 0.25f;
    attachment.rigidBodyEntityId      = disk.entityId;
    attachment.localAnchor            = {0.0f, 0.2f, 0.0f};
    attachment.localRotation          = Diligent::QuaternionF{0.0f, 0.0f, 0.38268343f, 0.9238795f};
    attachment.translationCompliance  = 0.01f;
    attachment.rotationCompliance     = 0.03f;

    const auto &authored = world.upsertStrandRigidAttachmentConstraint(attachment);
    if (authored.constraintId == physics::kInvalidStrandRigidAttachmentConstraintId)
    {
        CRESSIM_LOG_ERROR("Strand-rigid attachment id was not assigned.\n");
        return 1;
    }

    const auto &resolved = world.strandRigidAttachments();
    if (resolved.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Strand-rigid attachment did not resolve.\n");
        return 1;
    }

    const physics::StrandState *resolvedBackbone = world.tryGetStrand(backbone.entityId);
    if (resolvedBackbone == nullptr ||
        resolved[0].segmentIndex != resolvedBackbone->segmentOffset + 1u ||
        resolved[0].rigidBodyIndex != 0u ||
        std::abs(resolved[0].segmentT - 0.25f) > 1.0e-6f ||
        std::abs(resolved[0].translationCompliance - 0.01f) > 1.0e-6f ||
        std::abs(resolved[0].rotationCompliance - 0.03f) > 1.0e-6f ||
        std::abs(resolved[0].localAnchor.y - 0.2f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Resolved strand-rigid attachment payload is incorrect.\n");
        return 1;
    }

    const std::uint64_t topologyRevision = world.strandRigidAttachmentTopologyRevision();
    const std::uint64_t payloadRevision  = world.strandRigidAttachmentRevision();
    const std::uint64_t softTopologyRevision = world.softGpuTopologyRevision();
    physics::AuthoredStrandRigidAttachmentConstraintState updated = authored;
    updated.rotationCompliance = 0.06f;
    updated.enabled            = false;
    world.upsertStrandRigidAttachmentConstraint(updated);
    if (world.strandRigidAttachmentTopologyRevision() != topologyRevision ||
        world.strandRigidAttachmentRevision() == payloadRevision ||
        world.softGpuTopologyRevision() == softTopologyRevision ||
        !world.strandRigidAttachments().empty())
    {
        CRESSIM_LOG_ERROR("Strand-rigid attachment runtime payload update behaved incorrectly.\n");
        return 1;
    }

    updated.enabled           = true;
    updated.localSegmentIndex = 9u;
    world.upsertStrandRigidAttachmentConstraint(updated);
    if (world.strandRigidAttachmentTopologyRevision() == topologyRevision ||
        !world.strandRigidAttachments().empty())
    {
        CRESSIM_LOG_ERROR("Invalid strand-rigid attachment topology update did not rebuild.\n");
        return 1;
    }

    if (!world.removeStrandRigidAttachmentConstraint(authored.constraintId) ||
        !world.strandRigidAttachmentConstraintSnapshot().empty() ||
        !world.strandRigidAttachments().empty())
    {
        CRESSIM_LOG_ERROR("Removing strand-rigid attachment failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Strand-rigid attachment authoring checks passed.\n");
    return 0;
}
