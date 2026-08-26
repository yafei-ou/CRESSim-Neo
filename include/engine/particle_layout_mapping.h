#ifndef CRESSIM_NEO_ENGINE_PARTICLE_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_PARTICLE_LAYOUT_MAPPING_H

#include "common/id.h"
#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

/// @brief Prepared parallel arrays mapping particles and deformable owners to GPU slot layout.
struct ParticleLayoutMapping
{
    std::uint32_t particleCount  = 0u; ///< Number of particle slots.
    std::uint32_t softBodyCount  = 0u; ///< Number of soft-body owner slots.
    std::uint32_t fluidCount     = 0u; ///< Number of fluid owner slots.
    std::uint32_t strandCount    = 0u; ///< Number of strand owner slots.
    /// Prepared host-side particle/deformable slot-layout invalidation key produced by prepare().
    /// This describes when authored slot interpretation changes and is not the same as the live
    /// GPU custom-compute resource bindingGeneration exposed after uploadWorld().
    std::uint64_t layoutRevision = 0u;

    /// @brief Environment indices, indexed by particle slot in [0, particleCount).
    std::vector<std::uint32_t> environmentIndices;
    std::vector<std::uint32_t> particleKinds;         ///< Numeric particle-kind values.
    std::vector<std::uint32_t> ownerTypes;            ///< Numeric owner-type values.
    std::vector<std::uint32_t> ownerIndices;          ///< Owner slots within their owner type.
    std::vector<std::uint32_t> strandIds;             ///< Strand IDs where applicable.
    std::vector<std::uint32_t> strandOrders;          ///< Ordinal positions within strands.
    std::vector<std::uint32_t> strandRoles;           ///< Numeric strand-role values.
    std::vector<std::uint32_t> owningSoftBodyIndices; ///< Owning soft-body slots where applicable.
    std::vector<std::uint32_t> particleMaterialIndices; ///< Particle material slots.
    std::vector<std::uint32_t> fluidMaterialIndices;    ///< Fluid material slots.
    std::vector<std::uint32_t> phases;                  ///< Particle phase values.
    std::vector<std::uint32_t> collisionLayers;         ///< Collision layer masks.
    std::vector<std::uint32_t> collisionMasks;          ///< Collision filter masks.
    std::vector<std::uint32_t> adjacencyOffsets;        ///< Offsets into physics adjacency data.
    std::vector<std::uint32_t> adjacencyCounts;         ///< Adjacency counts.

    /// @brief Owning entity IDs, indexed by soft-body slot.
    std::vector<common::EntityId> softBodyEntityIds;
    std::vector<std::uint32_t> softBodyEnvironmentIndices; ///< Environment indices.
    std::vector<std::uint32_t> softBodyParticleOffsets;    ///< First particle slots.
    std::vector<std::uint32_t> softBodyParticleCounts;     ///< Particle counts.

    /// @brief Owning entity IDs, indexed by fluid slot.
    std::vector<common::EntityId> fluidEntityIds;
    std::vector<std::uint32_t> fluidEnvironmentIndices; ///< Environment indices.
    std::vector<std::uint32_t> fluidParticleOffsets;    ///< First particle slots.
    std::vector<std::uint32_t> fluidParticleCounts;     ///< Particle counts.

    /// @brief Owning entity IDs, indexed by strand slot.
    std::vector<common::EntityId> strandEntityIds;
    std::vector<std::uint32_t> strandEnvironmentIndices; ///< Environment indices.
    std::vector<std::uint32_t> strandParticleOffsets;    ///< First particle slots.
    std::vector<std::uint32_t> strandParticleCounts;     ///< Particle counts.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_PARTICLE_LAYOUT_MAPPING_H
