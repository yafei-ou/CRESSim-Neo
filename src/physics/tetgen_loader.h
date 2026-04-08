#ifndef CRESSIM_NEO_PHYSICS_TETGEN_LOADER_H
#define CRESSIM_NEO_PHYSICS_TETGEN_LOADER_H

#include "physics/physics_types.h"

#include <string>
#include <vector>

namespace cressim::neo::physics
{

struct TetMeshData
{
    std::vector<Diligent::float3> objectSpaceRestPositions;
    std::vector<std::uint32_t> tetVertexIndices;
};

bool loadTetGenFiles(const std::string &nodeFile, const std::string &eleFile, TetMeshData &outMesh,
                     std::string &errorMessage) noexcept;

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_TETGEN_LOADER_H
