#ifndef CRESSIM_NEO_PHYSICS_LOAD_PARTICLE_CLOUD_H
#define CRESSIM_NEO_PHYSICS_LOAD_PARTICLE_CLOUD_H

#include "physics/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <filesystem>
#include <string>
#include <vector>

/// @file load_particle_cloud.h
/// @brief Utilities for loading raw 3D particle coordinate arrays from binary data files.

namespace cressim::neo::physics
{

/// @brief Reads raw 3D particle coordinates from a binary `.bin` file on disk.
/// @param path File system path to the particle cloud binary file.
/// @param particles Output vector populated with 3D particle positions on success.
/// @param errorMessage Output error message description populated on failure.
/// @return True if particle cloud was read successfully, false otherwise.
CRESSIM_NEO_PHYSICS_API bool readParticleCloudBin(const std::filesystem::path &path,
                                                  std::vector<Diligent::float3> &particles,
                                                  std::string &errorMessage);

/// @brief Loads raw 3D particle coordinates from disk, throwing an exception on failure.
/// @param path File system path to the particle cloud binary file.
/// @return Vector of loaded 3D particle positions.
CRESSIM_NEO_PHYSICS_API std::vector<Diligent::float3> loadParticleCloud(
    const std::filesystem::path &path);

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_LOAD_PARTICLE_CLOUD_H
