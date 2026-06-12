#include "common/logger.h"
#include "physics/physics_world.h"

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::SoftBodyState softBody{};
    softBody.entityId                                = 9001u;
    softBody.environmentIndex                        = 3u;
    softBody.source.kind                             = physics::SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size                 = {1.0f, 1.0f, 1.0f};
    softBody.source.regularGrid.targetParticleSpacing = 0.5f;
    softBody.supportsSuturing                        = false;
    if (!world.upsertSoftBody(softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for suturing runtime update test.\n");
        return 1;
    }

    physics::StrandState strand{};
    strand.entityId         = 9002u;
    strand.environmentIndex = 3u;
    strand.suturingEnabled  = false;
    strand.pathNodeSpacing  = 0.18f;
    strand.restPositions = {
        {-0.2f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.2f, 0.0f, 0.0f},
    };
    if (!world.upsertStrand(strand))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for suturing runtime update test.\n");
        return 1;
    }

    if (!world.suturingPairs().empty() || world.reservedSuturingPathHeaderCount() != 0u ||
        world.reservedSuturingPathNodeCount() != 0u)
    {
        CRESSIM_LOG_ERROR("Unexpected suturing state before suturing is enabled.\n");
        return 1;
    }

    softBody.supportsSuturing = true;
    if (!world.upsertSoftBody(softBody))
    {
        CRESSIM_LOG_ERROR("Failed to enable soft-body suturing support at runtime.\n");
        return 1;
    }

    strand.suturingEnabled = true;
    if (!world.upsertStrand(strand))
    {
        CRESSIM_LOG_ERROR("Failed to enable strand suturing at runtime.\n");
        return 1;
    }

    const auto *authoredSoftBody = world.tryGetSoftBody(softBody.entityId);
    const auto *authoredStrand   = world.tryGetStrand(strand.entityId);
    if (authoredSoftBody == nullptr || !authoredSoftBody->supportsSuturing || authoredStrand == nullptr ||
        !authoredStrand->suturingEnabled)
    {
        CRESSIM_LOG_ERROR("Runtime suturing flags were not preserved in authored state.\n");
        return 1;
    }

    const auto &pairsAfterEnable = world.suturingPairs();
    if (pairsAfterEnable.size() != 1u || world.reservedSuturingPathHeaderCount() == 0u ||
        world.reservedSuturingPathNodeCount() == 0u)
    {
        CRESSIM_LOG_ERROR("Enabling suturing at runtime did not rebuild fallback suturing data.\n");
        return 1;
    }

    const physics::StrandSoftSuturingPair &fallbackPair = pairsAfterEnable.front();
    if (fallbackPair.softBodyIndex != 0u ||
        fallbackPair.suturingGroupId != 0u ||
        fallbackPair.pathNodeSpacing != strand.pathNodeSpacing)
    {
        CRESSIM_LOG_ERROR("Fallback suturing pair after runtime enable is incorrect.\n");
        return 1;
    }

    physics::AuthoredSuturingSequenceState invalidSequence{};
    invalidSequence.entries = {
        {strand.entityId, physics::AuthoredParticleReferenceType::StrandParticle, 0u},
        {999999u, physics::AuthoredParticleReferenceType::RigidProxyParticle, 0u},
    };
    invalidSequence.tipEntryIndex = 0u;
    world.upsertSuturingSequence(invalidSequence);

    const auto &pairsAfterInvalidSequence = world.suturingPairs();
    if (pairsAfterInvalidSequence.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Invalid suturing sequence should not suppress fallback suturing.\n");
        return 1;
    }

    const auto &particles = world.particles();
    if (authoredStrand->particleOffset >= particles.strandRoles.size() ||
        particles.strandRoles[authoredStrand->particleOffset] !=
            static_cast<std::uint32_t>(physics::ParticleStrandRole::NeedleTip))
    {
        CRESSIM_LOG_ERROR(
            "Invalid suturing sequence should not clear existing strand suturing metadata.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Suturing runtime update checks passed.\n");
    return 0;
}
