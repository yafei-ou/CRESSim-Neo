#ifndef CRESSIM_NEO_PHYSICS_SOFT_BODY_SHAPE_MATCHING_H
#define CRESSIM_NEO_PHYSICS_SOFT_BODY_SHAPE_MATCHING_H

#include "physics/export.h"
#include "physics/physics_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>
#include <vector>

namespace cressim::neo::physics
{

struct SoftBodyShapeMatchingCluster
{
    std::uint32_t seedParticle = 0u;
    std::vector<std::uint32_t> particles;
    std::vector<float> blendWeights;
    Diligent::float3 restCenter{0.0f, 0.0f, 0.0f};
};

struct SoftBodyShapeMatchingBuildResult
{
    std::vector<SoftBodyShapeMatchingCluster> clusters;
    std::vector<std::uint32_t> membershipCounts;
};

struct SoftBodyShapeMatchingPoseResult
{
    Diligent::float3 restCenter{0.0f, 0.0f, 0.0f};
    Diligent::float3 predictedCenter{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<Diligent::float3> goals;
};

CRESSIM_NEO_PHYSICS_API SoftBodyShapeMatchingPoseResult computeShapeMatchingClusterGoals(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<Diligent::float3> &predictedPositions,
    const std::vector<float> &fittingWeights,
    const std::vector<std::uint32_t> &clusterParticles) noexcept;

CRESSIM_NEO_PHYSICS_API std::vector<Diligent::float3> computeShapeMatchingClusterCorrections(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<Diligent::float3> &predictedPositions,
    const std::vector<float> &inverseMasses,
    const std::vector<float> &fittingWeights,
    const std::vector<std::uint32_t> &clusterParticles,
    float stiffness,
    float maximumCorrection = 0.0f) noexcept;

CRESSIM_NEO_PHYSICS_API SoftBodyShapeMatchingBuildResult buildOverlappingShapeMatchingClusters(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<std::vector<std::uint32_t>> &adjacencyLists,
    const SoftBodyShapeMatchingDesc &desc) noexcept;

CRESSIM_NEO_PHYSICS_API ShapeMatchingDataHost buildShapeMatchingGpuData(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<SoftBodyShapeMatchingCluster> &clusters,
    const SoftBodyShapeMatchingDesc &desc,
    std::uint32_t globalParticleOffset,
    std::uint32_t globalParticleCount) noexcept;

CRESSIM_NEO_PHYSICS_API std::vector<Diligent::float3> makeShapeMatchingReferenceCube(
    std::uint32_t sideCount = 5u, float spacing = 1.0f) noexcept;

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_SOFT_BODY_SHAPE_MATCHING_H
