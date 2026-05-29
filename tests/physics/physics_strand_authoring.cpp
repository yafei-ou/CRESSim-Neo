#include "physics/physics_world.h"
#include "common/logger.h"

int main()
{
    using namespace cressim::neo;

    physics::PhysicsWorld world;

    physics::StrandState strand{};
    strand.entityId             = 7001u;
    strand.environmentIndex     = 2u;
    strand.collisionLayer       = 0x8u;
    strand.collisionMask        = 0x24u;
    strand.particleMass         = 0.5f;
    strand.particleRadius       = 0.07f;
    strand.distanceCompliance   = 0.03f;
    strand.bendCompliance       = 0.015f;
    strand.selfCollisionEnabled = true;
    strand.restPositions = {
        {-0.5f, 0.0f, 0.0f},
        {0.0f, 0.2f, 0.0f},
        {0.5f, 0.0f, 0.0f},
    };
    strand.staticParticleIndices = {0u};

    if (!world.upsertStrand(strand))
    {
        CRESSIM_LOG_ERROR("Failed to author strand.\n");
        return 1;
    }

    if (world.strandCount() != 1u || world.softBodyCount() != 0u || world.fluidCount() != 0u)
    {
        CRESSIM_LOG_ERROR("Unexpected object counts after strand authoring.\n");
        return 1;
    }

    const physics::StrandState *authored = world.tryGetStrand(strand.entityId);
    if (authored == nullptr || authored->restPositions.size() != 3u)
    {
        CRESSIM_LOG_ERROR("Failed to retrieve authored strand state.\n");
        return 1;
    }

    const auto &particles = world.particles();
    const auto &constraints = world.distanceConstraints();
    const auto &bendConstraints = world.bendConstraints();
    if (particles.size() != 3u || constraints.size() != 2u || bendConstraints.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Unexpected strand-derived particle or constraint count.\n");
        return 1;
    }

    if (particles.ownerTypes[0] != static_cast<std::uint32_t>(physics::ParticleOwnerType::Strand) ||
        particles.deformableObjectKinds[0] !=
            static_cast<std::uint32_t>(physics::DeformableObjectKind::Strand) ||
        particles.strandIds[0] != 0u || particles.strandOrders[2] != 2u ||
        particles.strandRoles[1] != static_cast<std::uint32_t>(physics::ParticleStrandRole::None))
    {
        CRESSIM_LOG_ERROR("Strand particle metadata was not populated as expected.\n");
        return 1;
    }

    if (particles.positionsInvMass[0].w != 0.0f || particles.positionsInvMass[1].w <= 0.0f ||
        constraints[0].compliance != strand.distanceCompliance ||
        bendConstraints[0].compliance != strand.bendCompliance)
    {
        CRESSIM_LOG_ERROR("Strand particle masses or constraint properties are incorrect.\n");
        return 1;
    }

    if (authored->bendConstraintCount != 1u || authored->constraintCount != 2u)
    {
        CRESSIM_LOG_ERROR("Strand constraint counts were not populated as expected.\n");
        return 1;
    }

    if (!world.removeStrand(strand.entityId) || world.strandCount() != 0u ||
        world.tryGetStrand(strand.entityId) != nullptr)
    {
        CRESSIM_LOG_ERROR("Strand removal failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Strand authoring checks passed.\n");
    return 0;
}
