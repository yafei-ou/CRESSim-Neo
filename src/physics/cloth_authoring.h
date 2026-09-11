#ifndef CRESSIM_NEO_PHYSICS_CLOTH_AUTHORING_H
#define CRESSIM_NEO_PHYSICS_CLOTH_AUTHORING_H

#include "physics/physics_types.h"

#include <array>
#include <string>
#include <vector>

namespace cressim::neo::physics
{

struct CookedClothHinge
{
    std::uint32_t edgeVertex0     = 0u;
    std::uint32_t edgeVertex1     = 0u;
    std::uint32_t oppositeVertex0 = 0u;
    std::uint32_t oppositeVertex1 = 0u;
    float restAngle               = 0.0f;
};

struct CookedClothTopology
{
    std::vector<Diligent::float3> restPositions;
    std::vector<Diligent::uint3> triangles;
    std::vector<std::array<std::uint32_t, 2>> structuralPairs;
    std::vector<float> structuralRestLengths;
    std::vector<CookedClothHinge> bendHinges;
    std::vector<std::vector<std::uint32_t>> adjacencyLists;
    std::vector<std::uint32_t> staticParticleIndices;
};

bool cookClothTopology(const ClothState &state, CookedClothTopology &outTopology,
                       std::string &errorMessage) noexcept;

} // namespace cressim::neo::physics

#endif
