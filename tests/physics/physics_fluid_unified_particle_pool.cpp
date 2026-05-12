#include "physics/physics_world.h"
#include "common/logger.h"

#include <cmath>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::physics::FluidState;
using cressim::neo::physics::ParticleKind;
using cressim::neo::physics::ParticleOwnerType;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::physics::SoftBodyState;

SoftBodyState makeSoftBody(EntityId entityId)
{
    SoftBodyState state{};
    state.entityId = entityId;
    state.source.kind = SoftBodySourceKind::TetMesh;
    state.source.tetMesh.objectSpaceRestPositions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
    };
    state.source.tetMesh.tetVertexIndices = {0u, 1u, 2u, 3u};
    state.particleMass = 2.0f;
    state.particleRadius = 0.15f;
    state.selfCollisionEnabled = true;
    return state;
}

FluidState makeFluid(EntityId entityId)
{
    FluidState state{};
    state.entityId = entityId;
    state.source.kind = FluidSourceKind::RegularGrid;
    state.source.regularGrid.size = {0.2f, 0.2f, 0.2f};
    state.source.regularGrid.targetParticleSpacing = 0.2f;
    state.particleMass = 0.5f;
    state.particleRadius = 0.1f;
    state.material.viscosity = 0.15f;
    state.material.gravityScale = 0.9f;
    state.material.cohesion = 0.04f;
    state.material.surfaceTension = 0.02f;
    state.material.vorticityConfinement = 0.03f;
    state.material.cflCoefficient = 0.8f;
    state.restTransform.position = {2.0f, 0.0f, 0.0f};
    return state;
}

} // namespace

int main()
{
    PhysicsWorld world;

    const EntityId softEntity = 1001u;
    const EntityId fluidEntity = 1002u;

    if (!world.upsertSoftBody(makeSoftBody(softEntity)) || !world.upsertFluid(makeFluid(fluidEntity)))
    {
        CRESSIM_LOG_ERROR("Failed to author unified soft/fluid scene.\n");
        return 1;
    }

    world.ensureDerivedStateUpToDate();

    const auto &particles = world.particles();
    const auto &contactMaterials = world.particleContactMaterials();
    const auto &fluidMaterials = world.fluidMaterials();
    if (world.softBodyCount() != 1u || world.fluidCount() != 1u || particles.size() != 5u)
    {
        CRESSIM_LOG_ERROR("Unexpected unified particle pool sizes.\n");
        return 1;
    }

    std::uint32_t softCount = 0u;
    std::uint32_t fluidCount = 0u;
    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        if (particles.particleKinds[i] == static_cast<std::uint32_t>(ParticleKind::SoftSolid))
        {
            ++softCount;
            if (particles.ownerTypes[i] != static_cast<std::uint32_t>(ParticleOwnerType::SoftBody) ||
                particles.fluidMaterialIndices[i] != 0xffffffffu)
            {
                CRESSIM_LOG_ERROR("Soft particle metadata leaked fluid parameters.\n");
                return 1;
            }
        }
        else if (particles.particleKinds[i] == static_cast<std::uint32_t>(ParticleKind::Fluid))
        {
            ++fluidCount;
            const std::uint32_t fluidMaterialIndex = particles.fluidMaterialIndices[i];
            if (particles.ownerTypes[i] != static_cast<std::uint32_t>(ParticleOwnerType::FluidBody) ||
                fluidMaterialIndex >= fluidMaterials.size() ||
                fluidMaterials[fluidMaterialIndex].restDensity <= 0.0f ||
                fluidMaterials[fluidMaterialIndex].viscosityDerived <= 0.0f ||
                std::abs(fluidMaterials[fluidMaterialIndex].smoothingRadius - 0.4f) > 1.0e-5f ||
                fluidMaterials[fluidMaterialIndex].gravityScale != 0.9f ||
                fluidMaterials[fluidMaterialIndex].cohesionDerived <= 0.0f)
            {
                CRESSIM_LOG_ERROR("Fluid particle metadata was not propagated.\n");
                return 1;
            }
        }
        else
        {
            CRESSIM_LOG_ERROR("Unexpected particle kind in unified particle pool.\n");
            return 1;
        }
    }

    if (softCount != 4u || fluidCount != 1u)
    {
        CRESSIM_LOG_ERROR("Unified particle pool kind counts are incorrect.\n");
        return 1;
    }

    if (contactMaterials.size() != 1u || fluidMaterials.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Unified material tables were not deduplicated as expected.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Unified fluid/soft particle pool checks passed.\n");
    return 0;
}
