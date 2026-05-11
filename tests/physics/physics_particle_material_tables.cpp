#include "physics/physics_world.h"
#include "common/logger.h"

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::physics::FluidSourceKind;
using cressim::neo::physics::FluidState;
using cressim::neo::physics::ParticleKind;
using cressim::neo::physics::PhysicsWorld;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::physics::SoftBodyState;

SoftBodyState makeSoftBody(EntityId entityId, float offsetX)
{
    SoftBodyState state{};
    state.entityId = entityId;
    state.source.kind = SoftBodySourceKind::TetMesh;
    state.source.tetMesh.objectSpaceRestPositions = {
        {offsetX + 0.0f, 0.0f, 0.0f},
        {offsetX + 1.0f, 0.0f, 0.0f},
        {offsetX + 0.0f, 1.0f, 0.0f},
        {offsetX + 0.0f, 0.0f, 1.0f},
    };
    state.source.tetMesh.tetVertexIndices = {0u, 1u, 2u, 3u};
    state.material.contact.friction = 0.4f;
    state.material.contact.restitution = 0.1f;
    state.material.contact.damping = 0.05f;
    state.material.contact.staticFriction = 0.6f;
    state.particleRadius = 0.15f;
    return state;
}

FluidState makeFluid(EntityId entityId, float offsetX, float viscosity)
{
    FluidState state{};
    state.entityId = entityId;
    state.source.kind = FluidSourceKind::RegularGrid;
    state.source.regularGrid.size = {0.2f, 0.2f, 0.2f};
    state.source.regularGrid.targetParticleSpacing = 0.2f;
    state.restTransform.position = {offsetX, 0.0f, 0.0f};
    state.material.contact.friction = 0.4f;
    state.material.contact.restitution = 0.1f;
    state.material.contact.damping = 0.05f;
    state.material.contact.staticFriction = 0.6f;
    state.material.restDensity = 950.0f;
    state.material.viscosity = viscosity;
    state.material.smoothingRadius = 0.28f;
    return state;
}

} // namespace

int main()
{
    PhysicsWorld world;

    if (!world.upsertSoftBody(makeSoftBody(1001u, 0.0f)) ||
        !world.upsertSoftBody(makeSoftBody(1002u, 3.0f)) ||
        !world.upsertFluid(makeFluid(2001u, 6.0f, 0.02f)) ||
        !world.upsertFluid(makeFluid(2002u, 7.0f, 0.20f)))
    {
        CRESSIM_LOG_ERROR("Failed to author material-table test scene.\n");
        return 1;
    }

    FluidState incompatible = makeFluid(2003u, 8.0f, 0.08f);
    incompatible.material.restDensity = 1200.0f;
    if (world.upsertFluid(incompatible))
    {
        CRESSIM_LOG_ERROR("Incompatible fluid density model should have been rejected.\n");
        return 1;
    }

    world.ensureDerivedStateUpToDate();

    const auto &particles = world.particles();
    const auto &contactMaterials = world.particleContactMaterials();
    const auto &fluidMaterials = world.fluidMaterials();
    if (contactMaterials.size() != 1u || fluidMaterials.size() != 2u)
    {
        CRESSIM_LOG_ERROR("Material tables were not deduplicated as expected.\n");
        return 1;
    }

    const auto *softA = world.tryGetSoftBody(1001u);
    const auto *softB = world.tryGetSoftBody(1002u);
    const auto *fluidA = world.tryGetFluid(2001u);
    const auto *fluidB = world.tryGetFluid(2002u);
    if (softA == nullptr || softB == nullptr || fluidA == nullptr || fluidB == nullptr)
    {
        CRESSIM_LOG_ERROR("Authored bodies disappeared from world state.\n");
        return 1;
    }

    if (softA->contactMaterialIndex != softB->contactMaterialIndex ||
        softA->contactMaterialIndex != fluidA->contactMaterialIndex ||
        fluidA->contactMaterialIndex != fluidB->contactMaterialIndex)
    {
        CRESSIM_LOG_ERROR("Identical contact materials did not deduplicate.\n");
        return 1;
    }

    if (fluidA->fluidMaterialIndex == fluidB->fluidMaterialIndex)
    {
        CRESSIM_LOG_ERROR("Distinct fluid viscosities should produce distinct fluid materials.\n");
        return 1;
    }

    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(particles.size()); ++i)
    {
        if (particles.particleKinds[i] == static_cast<std::uint32_t>(ParticleKind::SoftSolid))
        {
            if (particles.fluidMaterialIndices[i] != 0xffffffffu ||
                particles.particleMaterialIndices[i] != softA->contactMaterialIndex)
            {
                CRESSIM_LOG_ERROR("Soft particle material indices are incorrect.\n");
                return 1;
            }
        }
        else if (particles.particleKinds[i] == static_cast<std::uint32_t>(ParticleKind::Fluid))
        {
            const std::uint32_t fluidMaterialIndex = particles.fluidMaterialIndices[i];
            if (fluidMaterialIndex >= fluidMaterials.size() ||
                particles.particleMaterialIndices[i] != fluidA->contactMaterialIndex)
            {
                CRESSIM_LOG_ERROR("Fluid particle material indices are incorrect.\n");
                return 1;
            }
        }
    }

    CRESSIM_LOG_INFO("Particle/fluid material table checks passed.\n");
    return 0;
}
