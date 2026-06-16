#include "common/logger.h"
#include "engine/world.h"

int main()
{
    using namespace cressim::neo;

    engine::World world;
    const common::EntityId diskEntity     = world.createEntity();
    const common::EntityId backboneEntity = world.createEntity();

    physics::RigidBodyState rigid{};
    rigid.entityId         = diskEntity;
    rigid.environmentIndex = 0u;
    rigid.bodyType         = physics::RigidBodyType::Dynamic;
    rigid.inverseMass      = 1.0f;
    world.physicsWorld().upsertRigidBody(rigid);

    physics::StrandState strand{};
    strand.entityId         = backboneEntity;
    strand.environmentIndex = 0u;
    strand.restPositions = {
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.5f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    if (!world.physicsWorld().upsertStrand(strand))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for engine strand-rigid attachment test.\n");
        return 1;
    }

    physics::AuthoredStrandRigidAttachmentConstraintState attachment{};
    attachment.strandEntityId    = backboneEntity;
    attachment.localSegmentIndex = 0u;
    attachment.segmentT          = 0.5f;
    attachment.rigidBodyEntityId = diskEntity;

    const auto &authored = world.upsertStrandRigidAttachmentConstraint(attachment);
    const auto *roundTripped = world.tryGetStrandRigidAttachmentConstraint(authored.constraintId);
    if (roundTripped == nullptr ||
        roundTripped->strandEntityId != backboneEntity ||
        roundTripped->rigidBodyEntityId != diskEntity ||
        roundTripped->localSegmentIndex != 0u)
    {
        CRESSIM_LOG_ERROR("Engine world strand-rigid attachment round-trip failed.\n");
        return 1;
    }

    if (!world.removeStrandRigidAttachmentConstraint(authored.constraintId) ||
        world.tryGetStrandRigidAttachmentConstraint(authored.constraintId) != nullptr)
    {
        CRESSIM_LOG_ERROR("Engine world strand-rigid attachment removal failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Engine strand-rigid attachment authoring checks passed.\n");
    return 0;
}
