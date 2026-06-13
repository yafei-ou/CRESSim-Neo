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
        {0.0f, 0.5f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    if (!world.physicsWorld().upsertStrand(strand))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for engine rigid-particle attachment test.\n");
        return 1;
    }

    physics::AuthoredRigidParticleAttachmentConstraintState attachment{};
    attachment.particle.entityId           = backboneEntity;
    attachment.particle.type               = physics::AuthoredParticleReferenceType::StrandParticle;
    attachment.particle.localParticleIndex = 1u;
    attachment.rigidBodyEntityId           = diskEntity;

    const auto &authored = world.upsertRigidParticleAttachmentConstraint(attachment);
    const auto *roundTripped = world.tryGetRigidParticleAttachmentConstraint(authored.constraintId);
    if (roundTripped == nullptr ||
        roundTripped->particle.entityId != backboneEntity ||
        roundTripped->rigidBodyEntityId != diskEntity)
    {
        CRESSIM_LOG_ERROR("Engine world rigid-particle attachment round-trip failed.\n");
        return 1;
    }

    if (!world.removeRigidParticleAttachmentConstraint(authored.constraintId) ||
        world.tryGetRigidParticleAttachmentConstraint(authored.constraintId) != nullptr)
    {
        CRESSIM_LOG_ERROR("Engine world rigid-particle attachment removal failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Engine rigid-particle attachment authoring checks passed.\n");
    return 0;
}
