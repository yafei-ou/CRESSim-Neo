#include "physics/physics_world.h"
#include "common/logger.h"

#include <cmath>

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
    strand.stretchShearCompliance = 0.03f;
    strand.bendCompliance       = 0.015f;
    strand.twistCompliance      = 0.01f;
    strand.distanceCompliance   = 0.002f;
    strand.selfCollisionEnabled = true;
    strand.suturingEnabled      = true;
    strand.pathNodeSpacing      = 0.2f;
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

    if (!authored->suturingEnabled || std::abs(authored->pathNodeSpacing - 0.2f) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Strand suturing metadata was not preserved in authored state.\n");
        return 1;
    }

    const auto &particles = world.particles();
    const auto &constraints = world.distanceConstraints();
    const auto &bendConstraints = world.bendConstraints();
    const auto &segments = world.strandSegments();
    const auto &joints = world.strandJoints();
    const auto &distanceConstraints = world.strandDistanceConstraints();
    if (particles.size() != 3u || constraints.size() != 0u || bendConstraints.size() != 0u ||
        segments.size() != 2u || joints.size() != 1u || distanceConstraints.size() != 2u)
    {
        CRESSIM_LOG_ERROR("Unexpected strand-derived particle or constraint count.\n");
        return 1;
    }

    if (particles.ownerTypes[0] != static_cast<std::uint32_t>(physics::ParticleOwnerType::Strand) ||
        particles.strandIds[0] != 0u || particles.strandOrders[2] != 2u ||
        particles.strandRoles[0] != static_cast<std::uint32_t>(physics::ParticleStrandRole::NeedleTip) ||
        particles.strandRoles[1] != static_cast<std::uint32_t>(physics::ParticleStrandRole::NeedleBody) ||
        particles.strandRoles[2] != static_cast<std::uint32_t>(physics::ParticleStrandRole::NeedleBody))
    {
        CRESSIM_LOG_ERROR("Strand particle metadata was not populated as expected.\n");
        return 1;
    }

    if (std::abs(particles.positionsInvMass[0].w) > 1.0e-6f ||
        particles.positionsInvMass[1].w <= 0.0f ||
        particles.positionsInvMass[2].w <= 0.0f ||
        std::abs(segments[0].stretchShearCompliance - strand.stretchShearCompliance) > 1.0e-6f ||
        std::abs(joints[0].bendCompliance - strand.bendCompliance) > 1.0e-6f ||
        std::abs(joints[0].twistCompliance - strand.twistCompliance) > 1.0e-6f ||
        std::abs(distanceConstraints[0].distanceCompliance - strand.distanceCompliance) > 1.0e-6f)
    {
        CRESSIM_LOG_ERROR("Strand particle masses or constraint properties are incorrect.\n");
        return 1;
    }

    if (authored->jointCount != 1u || authored->segmentCount != 2u)
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
