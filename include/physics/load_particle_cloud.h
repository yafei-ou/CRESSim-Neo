#ifndef CRESSIM_NEO_PHYSICS_LOAD_PARTICLE_CLOUD_H
#define CRESSIM_NEO_PHYSICS_LOAD_PARTICLE_CLOUD_H

#include "physics/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cressim::neo::physics
{

CRESSIM_NEO_PHYSICS_API bool readParticleCloudBin(const std::filesystem::path &path,
    std::vector<Diligent::float3> &particles, std::string &errorMessage);

CRESSIM_NEO_PHYSICS_API std::vector<Diligent::float3> loadParticleCloud(const std::filesystem::path &path);

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_LOAD_PARTICLE_CLOUD_H
