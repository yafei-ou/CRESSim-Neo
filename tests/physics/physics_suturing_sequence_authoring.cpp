#include "physics/physics_world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::SoftBodyState softBody{};
    softBody.entityId = 5001u;
    softBody.environmentIndex = 2u;
    softBody.supportsSuturing = true;
    softBody.source.kind = physics::SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size = {0.5f, 0.5f, 0.5f};
    softBody.source.regularGrid.targetParticleSpacing = 0.25f;
    if (!world.upsertSoftBody(softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for suturing sequence test.\n");
        return 1;
    }

    physics::RigidBodyState needle{};
    needle.entityId = 5002u;
    needle.environmentIndex = 2u;
    needle.bodyType = physics::RigidBodyType::Kinematic;
    needle.suturingEnabled = true;
    needle.needleTipProxyIndex = 0u;
    needle.proxyParticleRadius = 0.04f;
    needle.proxyParticleLocalPositions = {
        {-0.1f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.1f, 0.0f, 0.0f},
    };
    world.upsertRigidBody(needle);

    physics::StrandState thread{};
    thread.entityId = 5003u;
    thread.environmentIndex = 2u;
    thread.suturingEnabled = true;
    thread.pathNodeSpacing = 0.18f;
    thread.restPositions = {
        {0.1f, 0.0f, 0.0f},
        {0.24f, 0.0f, 0.0f},
        {0.38f, 0.0f, 0.0f},
    };
    if (!world.upsertStrand(thread))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for suturing sequence test.\n");
        return 1;
    }

    physics::AuthoredSuturingSequenceState sequence{};
    sequence.entries = {
        {needle.entityId, physics::AuthoredParticleReferenceType::RigidProxyParticle, 0u},
        {needle.entityId, physics::AuthoredParticleReferenceType::RigidProxyParticle, 1u},
        {needle.entityId, physics::AuthoredParticleReferenceType::RigidProxyParticle, 2u},
        {thread.entityId, physics::AuthoredParticleReferenceType::StrandParticle, 0u},
        {thread.entityId, physics::AuthoredParticleReferenceType::StrandParticle, 1u},
        {thread.entityId, physics::AuthoredParticleReferenceType::StrandParticle, 2u},
    };
    sequence.tipEntryIndex = 0u;
    sequence.pathNodeSpacing = 0.22f;

    const auto &authoredSequence = world.upsertSuturingSequence(sequence);
    if (authoredSequence.sequenceId == physics::kInvalidSuturingSequenceId ||
        world.suturingSequenceSnapshot().size() != 1u)
    {
        CRESSIM_LOG_ERROR("Suturing sequence id assignment failed.\n");
        return 1;
    }

    const physics::RigidBodyState *authoredNeedle = world.tryGetRigidBody(needle.entityId);
    const physics::StrandState *authoredThread = world.tryGetStrand(thread.entityId);
    if (authoredNeedle == nullptr || authoredThread == nullptr)
    {
        CRESSIM_LOG_ERROR("Failed to resolve rigid/strand state for suturing sequence test.\n");
        return 1;
    }

    const auto &particles = world.particles();
    const std::uint32_t expectedGroupId =
        static_cast<std::uint32_t>(world.strandSnapshot().size() + world.rigidBodySnapshot().size());
    for (std::uint32_t proxyIndex = 0u; proxyIndex < authoredNeedle->proxyParticleCount; ++proxyIndex)
    {
        const std::uint32_t particleIndex = authoredNeedle->proxyParticleOffset + proxyIndex;
        if (particles.strandIds[particleIndex] != expectedGroupId ||
            particles.strandOrders[particleIndex] != proxyIndex)
        {
            CRESSIM_LOG_ERROR("Needle proxy particle sequence metadata is incorrect.\n");
            return 1;
        }

        const physics::ParticleStrandRole expectedRole =
            proxyIndex == 0u ? physics::ParticleStrandRole::NeedleTip
                             : physics::ParticleStrandRole::NeedleBody;
        if (particles.strandRoles[particleIndex] != static_cast<std::uint32_t>(expectedRole))
        {
            CRESSIM_LOG_ERROR("Needle proxy particle sequence role is incorrect.\n");
            return 1;
        }
    }

    for (std::uint32_t localIndex = 0u; localIndex < authoredThread->particleCount; ++localIndex)
    {
        const std::uint32_t particleIndex = authoredThread->particleOffset + localIndex;
        if (particles.strandIds[particleIndex] != expectedGroupId ||
            particles.strandOrders[particleIndex] != authoredNeedle->proxyParticleCount + localIndex ||
            particles.strandRoles[particleIndex] !=
                static_cast<std::uint32_t>(physics::ParticleStrandRole::Thread))
        {
            CRESSIM_LOG_ERROR("Thread particle sequence metadata is incorrect.\n");
            return 1;
        }
    }

    const auto &pairs = world.suturingPairs();
    if (pairs.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Authored suturing sequence should yield exactly one suturing pair.\n");
        return 1;
    }

    const physics::StrandSoftSuturingPair &pair = pairs.front();
    if (pair.strandIndex != expectedGroupId ||
        pair.tipParticleIndex != authoredNeedle->proxyParticleOffset ||
        pair.pathNodeSpacing != 0.22f)
    {
        CRESSIM_LOG_ERROR("Resolved suturing pair for authored sequence is incorrect.\n");
        return 1;
    }

    if (!world.removeSuturingSequence(authoredSequence.sequenceId) ||
        !world.suturingSequenceSnapshot().empty())
    {
        CRESSIM_LOG_ERROR("Removing authored suturing sequence failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Suturing sequence authoring checks passed.\n");
    return 0;
}
