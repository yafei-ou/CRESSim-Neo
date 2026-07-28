#include "common/logger.h"
#include "physics/soft_body_shape_matching.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace
{

constexpr float kTolerance = 1.0e-4f;

float length(const Diligent::float3 &v)
{
    return std::sqrt(Diligent::dot(v, v));
}

float maxCorrectionLength(const std::vector<Diligent::float3> &corrections)
{
    float result = 0.0f;
    for (const Diligent::float3 &correction : corrections)
    {
        result = std::max(result, length(correction));
    }
    return result;
}

Diligent::float3 sumCorrections(const std::vector<Diligent::float3> &corrections)
{
    Diligent::float3 result{0.0f, 0.0f, 0.0f};
    for (const Diligent::float3 &correction : corrections)
    {
        result += correction;
    }
    return result;
}

std::vector<float> makeWeights(std::uint32_t count, float value = 1.0f)
{
    return std::vector<float>(count, value);
}

std::vector<std::uint32_t> allParticles(std::uint32_t count)
{
    std::vector<std::uint32_t> particles(count);
    std::iota(particles.begin(), particles.end(), 0u);
    return particles;
}

float quaternionAlignment(const Diligent::QuaternionF &lhs, const Diligent::QuaternionF &rhs)
{
    return std::abs(lhs.q.x * rhs.q.x + lhs.q.y * rhs.q.y + lhs.q.z * rhs.q.z +
                    lhs.q.w * rhs.q.w);
}

std::vector<std::vector<std::uint32_t>> makeCubeGraph(std::uint32_t sideCount)
{
    const auto index = [sideCount](std::uint32_t x, std::uint32_t y, std::uint32_t z)
    {
        return x + y * sideCount + z * sideCount * sideCount;
    };

    std::vector<std::vector<std::uint32_t>> adjacency(sideCount * sideCount * sideCount);
    for (std::uint32_t z = 0u; z < sideCount; ++z)
    {
        for (std::uint32_t y = 0u; y < sideCount; ++y)
        {
            for (std::uint32_t x = 0u; x < sideCount; ++x)
            {
                const std::uint32_t current = index(x, y, z);
                if (x + 1u < sideCount)
                {
                    adjacency[current].push_back(index(x + 1u, y, z));
                }
                if (x > 0u)
                {
                    adjacency[current].push_back(index(x - 1u, y, z));
                }
                if (y + 1u < sideCount)
                {
                    adjacency[current].push_back(index(x, y + 1u, z));
                }
                if (y > 0u)
                {
                    adjacency[current].push_back(index(x, y - 1u, z));
                }
                if (z + 1u < sideCount)
                {
                    adjacency[current].push_back(index(x, y, z + 1u));
                }
                if (z > 0u)
                {
                    adjacency[current].push_back(index(x, y, z - 1u));
                }
            }
        }
    }
    return adjacency;
}

bool testTranslationInvariance()
{
    using namespace cressim::neo::physics;
    const std::vector<Diligent::float3> rest = makeShapeMatchingReferenceCube();
    std::vector<Diligent::float3> predicted = rest;
    for (Diligent::float3 &position : predicted)
    {
        position += Diligent::float3{2.5f, -1.0f, 0.75f};
    }

    const std::vector<Diligent::float3> corrections = computeShapeMatchingClusterCorrections(
        rest, predicted, makeWeights(static_cast<std::uint32_t>(rest.size())),
        makeWeights(static_cast<std::uint32_t>(rest.size())),
        allParticles(static_cast<std::uint32_t>(rest.size())), 1.0f);

    if (maxCorrectionLength(corrections) > kTolerance)
    {
        CRESSIM_LOG_ERROR("Translation invariance failed; max correction was ",
                          maxCorrectionLength(corrections), ".\n");
        return false;
    }
    return true;
}

bool testRotationInvariance()
{
    using namespace cressim::neo::physics;
    const std::vector<Diligent::float3> rest = makeShapeMatchingReferenceCube();
    const Diligent::QuaternionF rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.4f, 0.8f, 0.2f}, 0.73f);

    std::vector<Diligent::float3> predicted;
    predicted.reserve(rest.size());
    for (const Diligent::float3 &position : rest)
    {
        predicted.push_back(rotation.RotateVector(position) +
                            Diligent::float3{0.5f, 1.25f, -0.2f});
    }

    const std::vector<std::uint32_t> cluster = allParticles(static_cast<std::uint32_t>(rest.size()));
    const SoftBodyShapeMatchingPoseResult pose = computeShapeMatchingClusterGoals(
        rest, predicted, makeWeights(static_cast<std::uint32_t>(rest.size())), cluster);
    const std::vector<Diligent::float3> corrections = computeShapeMatchingClusterCorrections(
        rest, predicted, makeWeights(static_cast<std::uint32_t>(rest.size())),
        makeWeights(static_cast<std::uint32_t>(rest.size())), cluster, 1.0f);

    if (maxCorrectionLength(corrections) > 5.0e-4f)
    {
        CRESSIM_LOG_ERROR("Rotation invariance failed; max correction was ",
                          maxCorrectionLength(corrections), ".\n");
        return false;
    }
    if (quaternionAlignment(pose.rotation, rotation) < 0.999f)
    {
        CRESSIM_LOG_ERROR("Rotation recovery failed; quaternion alignment was ",
                          quaternionAlignment(pose.rotation, rotation), ".\n");
        return false;
    }
    return true;
}

bool testLocalDeformation()
{
    using namespace cressim::neo::physics;
    const std::vector<Diligent::float3> rest = makeShapeMatchingReferenceCube();
    std::vector<Diligent::float3> predicted = rest;
    constexpr std::uint32_t displacedParticle = 62u;
    predicted[displacedParticle] += {0.85f, -0.35f, 0.2f};

    const std::vector<Diligent::float3> corrections = computeShapeMatchingClusterCorrections(
        rest, predicted, makeWeights(static_cast<std::uint32_t>(rest.size())),
        makeWeights(static_cast<std::uint32_t>(rest.size())),
        allParticles(static_cast<std::uint32_t>(rest.size())), 0.5f);

    if (Diligent::dot(corrections[displacedParticle],
                      rest[displacedParticle] - predicted[displacedParticle]) <= 0.0f)
    {
        CRESSIM_LOG_ERROR("Local deformation failed to restore the displaced particle.\n");
        return false;
    }
    if (maxCorrectionLength(corrections) <= 0.0f)
    {
        CRESSIM_LOG_ERROR("Local deformation produced no corrections.\n");
        return false;
    }
    if (length(sumCorrections(corrections)) >
        1.0e-3f * static_cast<float>(corrections.size()))
    {
        CRESSIM_LOG_ERROR("Local deformation introduced excessive center-of-mass drift.\n");
        return false;
    }
    return true;
}

bool testStaticParticles()
{
    using namespace cressim::neo::physics;
    const std::vector<Diligent::float3> rest = makeShapeMatchingReferenceCube();
    std::vector<Diligent::float3> predicted = rest;
    for (Diligent::float3 &position : predicted)
    {
        position += {0.25f, 0.0f, -0.1f};
    }
    predicted[0u] = rest[0u];

    std::vector<float> inverseMasses = makeWeights(static_cast<std::uint32_t>(rest.size()));
    inverseMasses[0u] = 0.0f;

    const std::vector<Diligent::float3> corrections = computeShapeMatchingClusterCorrections(
        rest, predicted, inverseMasses, makeWeights(static_cast<std::uint32_t>(rest.size())),
        allParticles(static_cast<std::uint32_t>(rest.size())), 1.0f);

    if (length(corrections[0u]) > kTolerance)
    {
        CRESSIM_LOG_ERROR("Static particle received a correction.\n");
        return false;
    }
    if (maxCorrectionLength(corrections) <= 1.0e-4f)
    {
        CRESSIM_LOG_ERROR("Static-particle test did not restore dynamic particles.\n");
        return false;
    }
    return true;
}

bool testOverlappingClusterConstruction()
{
    using namespace cressim::neo::physics;
    const std::vector<Diligent::float3> rest = makeShapeMatchingReferenceCube();
    SoftBodyShapeMatchingDesc desc{};
    desc.enabled = true;
    desc.targetClusterSize = 12u;
    desc.maximumClusterSize = 16u;
    desc.minimumMembershipsPerParticle = 3u;

    const SoftBodyShapeMatchingBuildResult result =
        buildOverlappingShapeMatchingClusters(rest, makeCubeGraph(5u), desc);
    if (result.clusters.empty())
    {
        CRESSIM_LOG_ERROR("Overlapping cluster construction produced no clusters.\n");
        return false;
    }
    if (*std::min_element(result.membershipCounts.begin(), result.membershipCounts.end()) <
        desc.minimumMembershipsPerParticle)
    {
        CRESSIM_LOG_ERROR("Overlapping cluster construction did not cover every particle.\n");
        return false;
    }

    std::vector<float> blendSums(rest.size(), 0.0f);
    for (const SoftBodyShapeMatchingCluster &cluster : result.clusters)
    {
        if (cluster.particles.size() < 4u || cluster.particles.size() > desc.maximumClusterSize)
        {
            CRESSIM_LOG_ERROR("Cluster size was outside the expected range.\n");
            return false;
        }
        for (std::uint32_t i = 0u; i < static_cast<std::uint32_t>(cluster.particles.size()); ++i)
        {
            blendSums[cluster.particles[i]] += cluster.blendWeights[i];
        }
    }
    for (const float sum : blendSums)
    {
        if (std::abs(sum - 1.0f) > 1.0e-3f)
        {
            CRESSIM_LOG_ERROR("Cluster blend weights were not normalized; sum was ", sum, ".\n");
            return false;
        }
    }
    return true;
}

bool testShapeMatchingGpuFlattening()
{
    using namespace cressim::neo::physics;
    const std::vector<Diligent::float3> rest = makeShapeMatchingReferenceCube();
    SoftBodyShapeMatchingDesc desc{};
    desc.enabled = true;
    desc.targetClusterSize = 12u;
    desc.maximumClusterSize = 16u;
    desc.minimumMembershipsPerParticle = 3u;
    desc.stiffnessPerPass = 0.2f;

    const SoftBodyShapeMatchingBuildResult built =
        buildOverlappingShapeMatchingClusters(rest, makeCubeGraph(5u), desc);
    const ShapeMatchingDataHost gpuData =
        buildShapeMatchingGpuData(rest, built.clusters, desc, 0u,
                                  static_cast<std::uint32_t>(rest.size()));

    if (gpuData.clusters.size() != built.clusters.size() ||
        gpuData.poses.size() != gpuData.clusters.size() ||
        gpuData.membershipClusterIndices.size() != gpuData.members.size() ||
        gpuData.particleMembershipRanges.size() != rest.size())
    {
        CRESSIM_LOG_ERROR("Flattened shape matching buffer sizes were inconsistent.\n");
        return false;
    }

    std::vector<float> blendSums(rest.size(), 0.0f);
    for (const ShapeClusterGPU &cluster : gpuData.clusters)
    {
        if ((cluster.flags & ShapeCluster_Active) == 0u ||
            std::abs(cluster.stiffness - desc.stiffnessPerPass) > kTolerance)
        {
            CRESSIM_LOG_ERROR("Flattened cluster metadata was not initialized correctly.\n");
            return false;
        }
        if (cluster.memberOffset + cluster.memberCount > gpuData.members.size())
        {
            CRESSIM_LOG_ERROR("Flattened cluster member range exceeded the member buffer.\n");
            return false;
        }

        const Diligent::float3 restCenter{cluster.restCenterAndMass.x,
                                          cluster.restCenterAndMass.y,
                                          cluster.restCenterAndMass.z};
        for (std::uint32_t i = 0u; i < cluster.memberCount; ++i)
        {
            const ShapeClusterMemberGPU &member = gpuData.members[cluster.memberOffset + i];
            if (member.particleIndex >= rest.size())
            {
                CRESSIM_LOG_ERROR("Flattened member referenced an invalid particle.\n");
                return false;
            }
            const std::uint32_t memberIndex = cluster.memberOffset + i;
            if (gpuData.membershipClusterIndices[memberIndex] >= gpuData.clusters.size())
            {
                CRESSIM_LOG_ERROR("Flattened membership-to-cluster index was invalid.\n");
                return false;
            }

            const Diligent::float3 expectedOffset = rest[member.particleIndex] - restCenter;
            const Diligent::float3 actualOffset{member.restOffset.x, member.restOffset.y,
                                                member.restOffset.z};
            if (length(actualOffset - expectedOffset) > kTolerance)
            {
                CRESSIM_LOG_ERROR("Flattened member rest offset was incorrect.\n");
                return false;
            }
            blendSums[member.particleIndex] += member.blendWeight;
        }
    }

    for (std::uint32_t particle = 0u; particle < static_cast<std::uint32_t>(rest.size());
         ++particle)
    {
        const ParticleShapeMembershipRangeGPU range =
            gpuData.particleMembershipRanges[particle];
        if (range.count < desc.minimumMembershipsPerParticle ||
            range.offset + range.count > gpuData.particleMembershipIndices.size())
        {
            CRESSIM_LOG_ERROR("Flattened reverse membership range was invalid.\n");
            return false;
        }
        for (std::uint32_t i = 0u; i < range.count; ++i)
        {
            const std::uint32_t memberIndex =
                gpuData.particleMembershipIndices[range.offset + i];
            if (memberIndex >= gpuData.members.size() ||
                gpuData.members[memberIndex].particleIndex != particle ||
                gpuData.membershipClusterIndices[memberIndex] >= gpuData.clusters.size())
            {
                CRESSIM_LOG_ERROR("Flattened reverse membership index was invalid.\n");
                return false;
            }
        }
        if (std::abs(blendSums[particle] - 1.0f) > 1.0e-3f)
        {
            CRESSIM_LOG_ERROR("Flattened blend weights were not normalized; sum was ",
                              blendSums[particle], ".\n");
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    if (!testTranslationInvariance() || !testRotationInvariance() || !testLocalDeformation() ||
        !testStaticParticles() || !testOverlappingClusterConstruction() ||
        !testShapeMatchingGpuFlattening())
    {
        return 1;
    }

    CRESSIM_LOG_INFO("CPU shape-matching reference checks passed.\n");
    return 0;
}
