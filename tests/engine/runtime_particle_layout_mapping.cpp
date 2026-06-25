#include "common/logger.h"
#include "engine/runtime.h"

#include <algorithm>

namespace
{

bool hasResource(const std::vector<cressim::neo::engine::CustomComputeResourceDesc> &resources,
                 const char *key)
{
    return std::any_of(resources.begin(), resources.end(),
                       [key](const cressim::neo::engine::CustomComputeResourceDesc &resource)
                       { return resource.key == key; });
}

} // namespace

int main()
{
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.sceneLayout.envCount           = 2u;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping runtime particle layout mapping test because runtime "
                            "initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();

    const common::EntityId softEntity = world.createEntity(0u);
    engine::TransformComponent softTransform{};
    softTransform.worldTransform.position = {0.0f, 0.0f, 0.0f};
    world.setTransform(softEntity, softTransform);
    engine::SoftBodyComponent softBody{};
    softBody.source.kind                          = physics::SoftBodySourceKind::RegularGrid;
    softBody.source.regularGrid.size              = {0.6f, 0.6f, 0.6f};
    softBody.source.regularGrid.targetParticleSpacing = 0.2f;
    softBody.particleRadius                       = 0.09f;
    if (!world.setSoftBody(softEntity, softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author soft body for runtime particle mapping test.");
        runtime.shutdown();
        return 1;
    }

    const common::EntityId fluidEntity = world.createEntity(1u);
    engine::TransformComponent fluidTransform{};
    fluidTransform.worldTransform.position = {0.0f, 0.25f, 2.0f};
    world.setTransform(fluidEntity, fluidTransform);
    engine::FluidComponent fluid{};
    fluid.source.kind                          = physics::FluidSourceKind::RegularGrid;
    fluid.source.regularGrid.size              = {0.6f, 0.6f, 0.6f};
    fluid.source.regularGrid.targetParticleSpacing = 0.15f;
    fluid.particleRadius                       = 0.07f;
    if (!world.setFluid(fluidEntity, fluid))
    {
        CRESSIM_LOG_ERROR("Failed to author fluid body for runtime particle mapping test.");
        runtime.shutdown();
        return 1;
    }

    const common::EntityId strandEntity = world.createEntity(1u);
    engine::StrandComponent strand{};
    strand.restPositions = {
        {-0.2f, 0.8f, 2.0f},
        {-0.1f, 0.7f, 2.0f},
        {0.0f, 0.6f, 2.0f},
        {0.1f, 0.5f, 2.0f},
    };
    strand.particleRadius = 0.05f;
    if (!world.setStrand(strandEntity, strand))
    {
        CRESSIM_LOG_ERROR("Failed to author strand for runtime particle mapping test.");
        runtime.shutdown();
        return 1;
    }

    runtime.prepare();

    engine::ParticleLayoutMapping mapping{};
    if (!runtime.tryGetPreparedParticleLayoutMapping(mapping))
    {
        CRESSIM_LOG_ERROR("Prepared particle layout mapping query failed.");
        runtime.shutdown();
        return 1;
    }

    if (mapping.layoutRevision == 0u || mapping.softBodyCount != 1u || mapping.fluidCount != 1u ||
        mapping.strandCount != 1u)
    {
        CRESSIM_LOG_ERROR("Unexpected prepared particle mapping counts or generation.");
        runtime.shutdown();
        return 1;
    }
    if (mapping.softBodyEntityIds.size() != 1u || mapping.softBodyEntityIds[0] != softEntity ||
        mapping.softBodyEnvironmentIndices[0] != 0u || mapping.softBodyParticleCounts[0] == 0u)
    {
        CRESSIM_LOG_ERROR("Soft-body prepared particle mapping metadata is incorrect.");
        runtime.shutdown();
        return 1;
    }
    if (mapping.fluidEntityIds.size() != 1u || mapping.fluidEntityIds[0] != fluidEntity ||
        mapping.fluidEnvironmentIndices[0] != 1u || mapping.fluidParticleCounts[0] == 0u)
    {
        CRESSIM_LOG_ERROR("Fluid prepared particle mapping metadata is incorrect.");
        runtime.shutdown();
        return 1;
    }
    if (mapping.strandEntityIds.size() != 1u || mapping.strandEntityIds[0] != strandEntity ||
        mapping.strandEnvironmentIndices[0] != 1u || mapping.strandParticleCounts[0] != 4u)
    {
        CRESSIM_LOG_ERROR("Strand prepared particle mapping metadata is incorrect.");
        runtime.shutdown();
        return 1;
    }
    if (mapping.environmentIndices.size() != mapping.particleCount ||
        mapping.particleKinds.size() != mapping.particleCount ||
        mapping.ownerTypes.size() != mapping.particleCount ||
        mapping.ownerIndices.size() != mapping.particleCount)
    {
        CRESSIM_LOG_ERROR("Prepared particle slot arrays do not match reported particle count.");
        runtime.shutdown();
        return 1;
    }

    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload runtime world for particle resource registry test.");
        runtime.shutdown();
        return 1;
    }

    const std::vector<engine::CustomComputeResourceDesc> resources =
        runtime.listCustomComputeResources();
    if (!hasResource(resources, "particle.positions_inv_mass") ||
        !hasResource(resources, "particle.velocities") ||
        !hasResource(resources, "particle.owner_indices") ||
        !hasResource(resources, "soft.edges") ||
        !hasResource(resources, "strand.segments") ||
        !hasResource(resources, "particle.fluid_material_indices"))
    {
        CRESSIM_LOG_ERROR("Expected particle/deformable custom-compute resource keys were missing.");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Prepared particle layout mapping and particle resource registry checks passed.");
    return 0;
}
