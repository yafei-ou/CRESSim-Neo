#ifndef CRESSIM_NEO_PHYSICS_LOAD_SURFACE_MESH_H
#define CRESSIM_NEO_PHYSICS_LOAD_SURFACE_MESH_H

#include "physics/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <filesystem>
#include <string>
#include <vector>

/// @file load_surface_mesh.h
/// @brief Utilities for parsing and loading triangle surface meshes from OBJ files.

namespace cressim::neo::physics
{

/// @brief Surface geometry representation comprising vertex positions, normals, and triangle
/// indices.
struct SurfaceMeshData
{
    std::vector<Diligent::float3>
        surfaceRestPositions;                     ///< Object-space rest positions of mesh vertices.
    std::vector<Diligent::float3> surfaceNormals; ///< Per-vertex surface normal vectors.
    std::vector<Diligent::uint3>
        surfaceTriangles; ///< Triangle index triplets indexing vertex arrays.
};

/// @brief Reads an OBJ format triangle surface mesh from disk.
/// @param path File system path to the `.obj` file.
/// @param mesh Output SurfaceMeshData structure populated on success.
/// @param errorMessage Output error message description populated on failure.
/// @return True if file was read and parsed successfully, false otherwise.
CRESSIM_NEO_PHYSICS_API bool readObjSurfaceMesh(const std::filesystem::path &path,
                                                SurfaceMeshData &mesh, std::string &errorMessage);

/// @brief Loads an OBJ format triangle surface mesh from disk.
/// @param path File system path to the `.obj` file.
/// @return Loaded mesh, or an empty mesh after logging a read/parse failure.
CRESSIM_NEO_PHYSICS_API SurfaceMeshData loadObjSurfaceMesh(const std::filesystem::path &path);

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_LOAD_SURFACE_MESH_H
