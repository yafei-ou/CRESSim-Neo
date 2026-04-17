#ifndef CRESSIM_NEO_PHYSICS_SOFT_BODY_AUTHORING_H
#define CRESSIM_NEO_PHYSICS_SOFT_BODY_AUTHORING_H

#include "physics/physics_types.h"
#include "physics/tetgen_loader.h"

#include <array>
#include <string>
#include <vector>

namespace cressim::neo::physics
{

struct ResolvedSoftBodyTopology
{
    std::vector<Diligent::float3> restPositions;
    std::vector<std::array<std::uint32_t, 2>> edges;
    std::vector<std::array<std::uint32_t, 4>> tets;
    std::vector<Diligent::uint3> boundaryFaces;
    std::vector<std::vector<std::uint32_t>> adjacencyLists;
    std::vector<std::uint32_t> staticParticleIndices;
};

bool resolveSoftBodyTopology(const SoftBodyState &state, const TetMeshData *tetGenCache,
                             ResolvedSoftBodyTopology &outTopology,
                             std::string &errorMessage) noexcept;

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_SOFT_BODY_AUTHORING_H
