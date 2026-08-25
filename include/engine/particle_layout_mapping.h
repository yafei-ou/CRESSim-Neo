#ifndef CRESSIM_NEO_ENGINE_PARTICLE_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_PARTICLE_LAYOUT_MAPPING_H

#include "common/id.h"
#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

struct ParticleLayoutMapping
{
    std::uint32_t particleCount  = 0u;
    std::uint32_t softBodyCount  = 0u;
    std::uint32_t fluidCount     = 0u;
    std::uint32_t strandCount    = 0u;
    /// Prepared host-side particle/deformable slot-layout invalidation key produced by prepare().
    /// This describes when authored slot interpretation changes and is not the same as the live
    /// GPU custom-compute resource bindingGeneration exposed after uploadWorld().
    std::uint64_t layoutRevision = 0u;

    std::vector<std::uint32_t> environmentIndices;
    std::vector<std::uint32_t> particleKinds;
    std::vector<std::uint32_t> ownerTypes;
    std::vector<std::uint32_t> ownerIndices;
    std::vector<std::uint32_t> strandIds;
    std::vector<std::uint32_t> strandOrders;
    std::vector<std::uint32_t> strandRoles;
    std::vector<std::uint32_t> owningSoftBodyIndices;
    std::vector<std::uint32_t> particleMaterialIndices;
    std::vector<std::uint32_t> fluidMaterialIndices;
    std::vector<std::uint32_t> phases;
    std::vector<std::uint32_t> collisionLayers;
    std::vector<std::uint32_t> collisionMasks;
    std::vector<std::uint32_t> adjacencyOffsets;
    std::vector<std::uint32_t> adjacencyCounts;

    std::vector<common::EntityId> softBodyEntityIds;
    std::vector<std::uint32_t> softBodyEnvironmentIndices;
    std::vector<std::uint32_t> softBodyParticleOffsets;
    std::vector<std::uint32_t> softBodyParticleCounts;

    std::vector<common::EntityId> fluidEntityIds;
    std::vector<std::uint32_t> fluidEnvironmentIndices;
    std::vector<std::uint32_t> fluidParticleOffsets;
    std::vector<std::uint32_t> fluidParticleCounts;

    std::vector<common::EntityId> strandEntityIds;
    std::vector<std::uint32_t> strandEnvironmentIndices;
    std::vector<std::uint32_t> strandParticleOffsets;
    std::vector<std::uint32_t> strandParticleCounts;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_PARTICLE_LAYOUT_MAPPING_H
