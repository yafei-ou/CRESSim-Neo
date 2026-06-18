#ifndef CRESSIM_NEO_PHYSICS_LOAD_SURFACE_MESH_H
#define CRESSIM_NEO_PHYSICS_LOAD_SURFACE_MESH_H

#include "physics/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cressim::neo::physics
{

struct SurfaceMeshData
{
    std::vector<Diligent::float3> surfaceRestPositions;
    std::vector<Diligent::float3> surfaceNormals;
    std::vector<Diligent::uint3> surfaceTriangles;
};

CRESSIM_NEO_PHYSICS_API bool readObjSurfaceMesh(const std::filesystem::path &path,
                                                SurfaceMeshData &mesh, std::string &errorMessage);

CRESSIM_NEO_PHYSICS_API SurfaceMeshData loadObjSurfaceMesh(const std::filesystem::path &path);

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_LOAD_SURFACE_MESH_H
