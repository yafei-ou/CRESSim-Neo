#include "physics/soft_body_shape_matching.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>

namespace cressim::neo::physics
{

namespace
{

struct Mat3
{
    float m[3][3]{};
};

float lengthSq(const Diligent::float3 &v)
{
    return Diligent::dot(v, v);
}

Diligent::float3 safeNormalize(const Diligent::float3 &v)
{
    const float lenSq = lengthSq(v);
    if (lenSq <= 1.0e-20f)
    {
        return {0.0f, 0.0f, 0.0f};
    }
    return v / std::sqrt(lenSq);
}

Mat3 transpose(const Mat3 &a)
{
    Mat3 result{};
    for (std::uint32_t r = 0u; r < 3u; ++r)
    {
        for (std::uint32_t c = 0u; c < 3u; ++c)
        {
            result.m[r][c] = a.m[c][r];
        }
    }
    return result;
}

float determinant(const Mat3 &a)
{
    return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
           a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
           a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}

Mat3 inverse(const Mat3 &a)
{
    Mat3 result{};
    const float det = determinant(a);
    if (std::abs(det) <= 1.0e-12f)
    {
        return result;
    }

    const float invDet = 1.0f / det;
    result.m[0][0] = (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) * invDet;
    result.m[0][1] = (a.m[0][2] * a.m[2][1] - a.m[0][1] * a.m[2][2]) * invDet;
    result.m[0][2] = (a.m[0][1] * a.m[1][2] - a.m[0][2] * a.m[1][1]) * invDet;
    result.m[1][0] = (a.m[1][2] * a.m[2][0] - a.m[1][0] * a.m[2][2]) * invDet;
    result.m[1][1] = (a.m[0][0] * a.m[2][2] - a.m[0][2] * a.m[2][0]) * invDet;
    result.m[1][2] = (a.m[0][2] * a.m[1][0] - a.m[0][0] * a.m[1][2]) * invDet;
    result.m[2][0] = (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]) * invDet;
    result.m[2][1] = (a.m[0][1] * a.m[2][0] - a.m[0][0] * a.m[2][1]) * invDet;
    result.m[2][2] = (a.m[0][0] * a.m[1][1] - a.m[0][1] * a.m[1][0]) * invDet;
    return result;
}

Mat3 add(const Mat3 &a, const Mat3 &b, const float scale)
{
    Mat3 result{};
    for (std::uint32_t r = 0u; r < 3u; ++r)
    {
        for (std::uint32_t c = 0u; c < 3u; ++c)
        {
            result.m[r][c] = (a.m[r][c] + b.m[r][c]) * scale;
        }
    }
    return result;
}

Mat3 makeIdentity()
{
    Mat3 result{};
    result.m[0][0] = 1.0f;
    result.m[1][1] = 1.0f;
    result.m[2][2] = 1.0f;
    return result;
}

Mat3 outerProduct(const Diligent::float3 &a, const Diligent::float3 &b, const float weight)
{
    Mat3 result{};
    result.m[0][0] = weight * a.x * b.x;
    result.m[0][1] = weight * a.x * b.y;
    result.m[0][2] = weight * a.x * b.z;
    result.m[1][0] = weight * a.y * b.x;
    result.m[1][1] = weight * a.y * b.y;
    result.m[1][2] = weight * a.y * b.z;
    result.m[2][0] = weight * a.z * b.x;
    result.m[2][1] = weight * a.z * b.y;
    result.m[2][2] = weight * a.z * b.z;
    return result;
}

void accumulate(Mat3 &dst, const Mat3 &src)
{
    for (std::uint32_t r = 0u; r < 3u; ++r)
    {
        for (std::uint32_t c = 0u; c < 3u; ++c)
        {
            dst.m[r][c] += src.m[r][c];
        }
    }
}

float frobeniusNorm(const Mat3 &a)
{
    float sum = 0.0f;
    for (std::uint32_t r = 0u; r < 3u; ++r)
    {
        for (std::uint32_t c = 0u; c < 3u; ++c)
        {
            sum += a.m[r][c] * a.m[r][c];
        }
    }
    return std::sqrt(sum);
}

Diligent::float3 column(const Mat3 &a, const std::uint32_t c)
{
    return {a.m[0][c], a.m[1][c], a.m[2][c]};
}

void setColumn(Mat3 &a, const std::uint32_t c, const Diligent::float3 &v)
{
    a.m[0][c] = v.x;
    a.m[1][c] = v.y;
    a.m[2][c] = v.z;
}

Mat3 orthonormalizeProper(Mat3 a)
{
    Diligent::float3 c0 = safeNormalize(column(a, 0u));
    Diligent::float3 c1 = column(a, 1u) - c0 * Diligent::dot(c0, column(a, 1u));
    c1 = safeNormalize(c1);
    Diligent::float3 c2 = Diligent::cross(c0, c1);
    if (lengthSq(c0) <= 1.0e-12f || lengthSq(c1) <= 1.0e-12f || lengthSq(c2) <= 1.0e-12f)
    {
        return makeIdentity();
    }
    if (Diligent::dot(c2, column(a, 2u)) < 0.0f)
    {
        c1 = c1 * -1.0f;
        c2 = c2 * -1.0f;
    }
    setColumn(a, 0u, c0);
    setColumn(a, 1u, c1);
    setColumn(a, 2u, safeNormalize(c2));
    return a;
}

Diligent::QuaternionF quaternionFromRotationMatrix(const Mat3 &r)
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
    const float trace = r.m[0][0] + r.m[1][1] + r.m[2][2];
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        w = 0.25f * s;
        x = (r.m[2][1] - r.m[1][2]) / s;
        y = (r.m[0][2] - r.m[2][0]) / s;
        z = (r.m[1][0] - r.m[0][1]) / s;
    }
    else if (r.m[0][0] > r.m[1][1] && r.m[0][0] > r.m[2][2])
    {
        const float s = std::sqrt(1.0f + r.m[0][0] - r.m[1][1] - r.m[2][2]) * 2.0f;
        w = (r.m[2][1] - r.m[1][2]) / s;
        x = 0.25f * s;
        y = (r.m[0][1] + r.m[1][0]) / s;
        z = (r.m[0][2] + r.m[2][0]) / s;
    }
    else if (r.m[1][1] > r.m[2][2])
    {
        const float s = std::sqrt(1.0f + r.m[1][1] - r.m[0][0] - r.m[2][2]) * 2.0f;
        w = (r.m[0][2] - r.m[2][0]) / s;
        x = (r.m[0][1] + r.m[1][0]) / s;
        y = 0.25f * s;
        z = (r.m[1][2] + r.m[2][1]) / s;
    }
    else
    {
        const float s = std::sqrt(1.0f + r.m[2][2] - r.m[0][0] - r.m[1][1]) * 2.0f;
        w = (r.m[1][0] - r.m[0][1]) / s;
        x = (r.m[0][2] + r.m[2][0]) / s;
        y = (r.m[1][2] + r.m[2][1]) / s;
        z = 0.25f * s;
    }

    const float len = std::sqrt(x * x + y * y + z * z + w * w);
    if (len <= 1.0e-12f)
    {
        return {0.0f, 0.0f, 0.0f, 1.0f};
    }
    return {x / len, y / len, z / len, w / len};
}

Mat3 closestProperRotation(const Mat3 &apq)
{
    if (frobeniusNorm(apq) <= 1.0e-12f)
    {
        return makeIdentity();
    }

    Mat3 r = apq;
    for (std::uint32_t i = 0u; i < 12u; ++i)
    {
        const Mat3 invTrans = transpose(inverse(r));
        if (frobeniusNorm(invTrans) <= 1.0e-12f)
        {
            break;
        }
        r = add(r, invTrans, 0.5f);
    }

    r = orthonormalizeProper(r);
    if (determinant(r) < 0.0f)
    {
        setColumn(r, 2u, column(r, 2u) * -1.0f);
    }
    return r;
}

std::array<float, 3> symmetricEigenvalues(Mat3 a)
{
    for (std::uint32_t iteration = 0u; iteration < 24u; ++iteration)
    {
        std::uint32_t p = 0u;
        std::uint32_t q = 1u;
        float maxOffDiag = std::abs(a.m[p][q]);
        for (std::uint32_t r = 0u; r < 3u; ++r)
        {
            for (std::uint32_t c = r + 1u; c < 3u; ++c)
            {
                const float value = std::abs(a.m[r][c]);
                if (value > maxOffDiag)
                {
                    maxOffDiag = value;
                    p = r;
                    q = c;
                }
            }
        }
        if (maxOffDiag <= 1.0e-12f)
        {
            break;
        }

        const float theta = 0.5f * std::atan2(2.0f * a.m[p][q], a.m[q][q] - a.m[p][p]);
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        const float app = c * c * a.m[p][p] - 2.0f * s * c * a.m[p][q] + s * s * a.m[q][q];
        const float aqq = s * s * a.m[p][p] + 2.0f * s * c * a.m[p][q] + c * c * a.m[q][q];
        a.m[p][p] = app;
        a.m[q][q] = aqq;
        a.m[p][q] = 0.0f;
        a.m[q][p] = 0.0f;
        for (std::uint32_t k = 0u; k < 3u; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }
            const float aik = a.m[p][k];
            const float aqk = a.m[q][k];
            a.m[p][k] = c * aik - s * aqk;
            a.m[k][p] = a.m[p][k];
            a.m[q][k] = s * aik + c * aqk;
            a.m[k][q] = a.m[q][k];
        }
    }

    std::array<float, 3> values{a.m[0][0], a.m[1][1], a.m[2][2]};
    std::sort(values.begin(), values.end());
    return values;
}

Diligent::float3 weightedCenter(const std::vector<Diligent::float3> &positions,
                                const std::vector<float> &weights,
                                const std::vector<std::uint32_t> &members)
{
    Diligent::float3 center{0.0f, 0.0f, 0.0f};
    float totalWeight = 0.0f;
    for (const std::uint32_t particle : members)
    {
        if (particle >= positions.size())
        {
            continue;
        }
        const float weight = particle < weights.size() ? std::max(weights[particle], 0.0f) : 1.0f;
        center += positions[particle] * weight;
        totalWeight += weight;
    }
    return totalWeight > 0.0f ? center / totalWeight : center;
}

bool clusterHasSufficientExtent(const std::vector<Diligent::float3> &restPositions,
                                const std::vector<std::uint32_t> &members)
{
    if (members.size() < 4u)
    {
        return false;
    }

    Diligent::float3 center{0.0f, 0.0f, 0.0f};
    for (const std::uint32_t particle : members)
    {
        center += restPositions[particle];
    }
    center = center / static_cast<float>(members.size());

    Mat3 covariance{};
    float maxExtentSq = 0.0f;
    for (const std::uint32_t particle : members)
    {
        const Diligent::float3 q = restPositions[particle] - center;
        maxExtentSq = std::max(maxExtentSq, lengthSq(q));
        accumulate(covariance, outerProduct(q, q, 1.0f));
    }
    if (maxExtentSq <= 1.0e-12f)
    {
        return false;
    }

    const std::array<float, 3> values = symmetricEigenvalues(covariance);
    const float lambdaMax = std::max(values[2], 0.0f);
    const float lambdaMin = std::max(values[0], 0.0f);
    return lambdaMax > 0.0f && lambdaMin / lambdaMax > 1.0e-4f;
}

bool inducedGraphIsConnected(const std::vector<std::vector<std::uint32_t>> &adjacencyLists,
                             const std::vector<std::uint32_t> &members)
{
    if (members.empty())
    {
        return false;
    }

    std::unordered_set<std::uint32_t> memberSet(members.begin(), members.end());
    std::vector<std::uint32_t> stack{members.front()};
    std::unordered_set<std::uint32_t> visited;
    visited.reserve(members.size());
    while (!stack.empty())
    {
        const std::uint32_t particle = stack.back();
        stack.pop_back();
        if (!visited.insert(particle).second || particle >= adjacencyLists.size())
        {
            continue;
        }
        for (const std::uint32_t neighbor : adjacencyLists[particle])
        {
            if (memberSet.find(neighbor) != memberSet.end() &&
                visited.find(neighbor) == visited.end())
            {
                stack.push_back(neighbor);
            }
        }
    }
    return visited.size() == members.size();
}

std::vector<std::uint32_t> collectConnectedGraphNeighborhood(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<std::vector<std::uint32_t>> &adjacencyLists,
    const std::uint32_t seed,
    const std::uint32_t targetSize,
    const std::uint32_t maximumSize)
{
    struct Candidate
    {
        float distanceSq = 0.0f;
        std::uint32_t particle = 0u;
        bool operator>(const Candidate &rhs) const
        {
            if (distanceSq != rhs.distanceSq)
            {
                return distanceSq > rhs.distanceSq;
            }
            return particle > rhs.particle;
        }
    };

    std::vector<std::uint32_t> members;
    if (seed >= restPositions.size() || seed >= adjacencyLists.size())
    {
        return members;
    }
    members.reserve(maximumSize);

    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
    std::vector<std::uint8_t> seen(restPositions.size(), 0u);
    const auto push = [&](const std::uint32_t particle)
    {
        if (particle >= restPositions.size() || seen[particle] != 0u)
        {
            return;
        }
        seen[particle] = 1u;
        queue.push({lengthSq(restPositions[particle] - restPositions[seed]), particle});
    };

    push(seed);
    while (!queue.empty() && members.size() < maximumSize)
    {
        const Candidate candidate = queue.top();
        queue.pop();
        members.push_back(candidate.particle);

        if (members.size() >= targetSize && clusterHasSufficientExtent(restPositions, members))
        {
            break;
        }

        if (candidate.particle < adjacencyLists.size())
        {
            for (const std::uint32_t neighbor : adjacencyLists[candidate.particle])
            {
                push(neighbor);
            }
        }
    }

    return members;
}

std::uint32_t minimumMembershipCount(const std::vector<std::uint32_t> &membershipCounts)
{
    return membershipCounts.empty()
               ? 0u
               : *std::min_element(membershipCounts.begin(), membershipCounts.end());
}

std::uint32_t leastCoveredParticle(const std::vector<std::uint32_t> &membershipCounts,
                                   const std::vector<std::uint8_t> &rejectedSeeds,
                                   const std::uint32_t minimumMemberships)
{
    std::uint32_t bestParticle = static_cast<std::uint32_t>(membershipCounts.size());
    std::uint32_t bestCount = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t particle = 0u; particle < membershipCounts.size(); ++particle)
    {
        if (membershipCounts[particle] >= minimumMemberships ||
            (particle < rejectedSeeds.size() && rejectedSeeds[particle] != 0u))
        {
            continue;
        }
        if (membershipCounts[particle] < bestCount)
        {
            bestCount = membershipCounts[particle];
            bestParticle = particle;
        }
    }
    return bestParticle;
}

} // namespace

SoftBodyShapeMatchingPoseResult computeShapeMatchingClusterGoals(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<Diligent::float3> &predictedPositions,
    const std::vector<float> &fittingWeights,
    const std::vector<std::uint32_t> &clusterParticles) noexcept
{
    SoftBodyShapeMatchingPoseResult result{};
    if (clusterParticles.empty())
    {
        return result;
    }

    result.restCenter = weightedCenter(restPositions, fittingWeights, clusterParticles);
    result.predictedCenter = weightedCenter(predictedPositions, fittingWeights, clusterParticles);

    Mat3 apq{};
    for (const std::uint32_t particle : clusterParticles)
    {
        if (particle >= restPositions.size() || particle >= predictedPositions.size())
        {
            continue;
        }
        const float weight =
            particle < fittingWeights.size() ? std::max(fittingWeights[particle], 0.0f) : 1.0f;
        const Diligent::float3 q = restPositions[particle] - result.restCenter;
        const Diligent::float3 p = predictedPositions[particle] - result.predictedCenter;
        accumulate(apq, outerProduct(p, q, weight));
    }

    const Mat3 rotationMatrix = closestProperRotation(apq);
    result.rotation = quaternionFromRotationMatrix(rotationMatrix);
    result.goals.reserve(clusterParticles.size());
    for (const std::uint32_t particle : clusterParticles)
    {
        if (particle >= restPositions.size())
        {
            result.goals.push_back(result.predictedCenter);
            continue;
        }
        result.goals.push_back(
            result.predictedCenter +
            result.rotation.RotateVector(restPositions[particle] - result.restCenter));
    }
    return result;
}

std::vector<Diligent::float3> computeShapeMatchingClusterCorrections(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<Diligent::float3> &predictedPositions,
    const std::vector<float> &inverseMasses,
    const std::vector<float> &fittingWeights,
    const std::vector<std::uint32_t> &clusterParticles,
    const float stiffness,
    const float maximumCorrection) noexcept
{
    std::vector<Diligent::float3> corrections(predictedPositions.size(),
                                              Diligent::float3{0.0f, 0.0f, 0.0f});
    const SoftBodyShapeMatchingPoseResult pose = computeShapeMatchingClusterGoals(
        restPositions, predictedPositions, fittingWeights, clusterParticles);
    const float clampedStiffness = std::clamp(stiffness, 0.0f, 1.0f);

    for (std::uint32_t memberIndex = 0u;
         memberIndex < static_cast<std::uint32_t>(clusterParticles.size()) &&
         memberIndex < static_cast<std::uint32_t>(pose.goals.size());
         ++memberIndex)
    {
        const std::uint32_t particle = clusterParticles[memberIndex];
        if (particle >= predictedPositions.size())
        {
            continue;
        }
        const float invMass = particle < inverseMasses.size() ? inverseMasses[particle] : 1.0f;
        if (invMass <= 0.0f)
        {
            continue;
        }

        Diligent::float3 correction =
            (pose.goals[memberIndex] - predictedPositions[particle]) * clampedStiffness;
        if (maximumCorrection > 0.0f)
        {
            const float correctionLenSq = lengthSq(correction);
            const float maxLenSq = maximumCorrection * maximumCorrection;
            if (correctionLenSq > maxLenSq && correctionLenSq > 0.0f)
            {
                correction = correction * (maximumCorrection / std::sqrt(correctionLenSq));
            }
        }
        corrections[particle] = correction;
    }
    return corrections;
}

SoftBodyShapeMatchingBuildResult buildOverlappingShapeMatchingClusters(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<std::vector<std::uint32_t>> &adjacencyLists,
    const SoftBodyShapeMatchingDesc &desc) noexcept
{
    SoftBodyShapeMatchingBuildResult result{};
    const std::uint32_t particleCount = static_cast<std::uint32_t>(restPositions.size());
    result.membershipCounts.assign(particleCount, 0u);
    if (!desc.enabled || particleCount == 0u || adjacencyLists.size() < particleCount)
    {
        return result;
    }

    const std::uint32_t targetSize =
        std::clamp(desc.targetClusterSize, 4u,
                   std::min<std::uint32_t>(16u, std::max<std::uint32_t>(4u, particleCount)));
    const std::uint32_t maximumSize =
        std::clamp(desc.maximumClusterSize, targetSize,
                   std::min<std::uint32_t>(16u, std::max(targetSize, particleCount)));
    const std::uint32_t minimumMemberships =
        std::max<std::uint32_t>(1u, desc.minimumMembershipsPerParticle);
    const std::uint32_t maxAttempts = particleCount * minimumMemberships * 8u + particleCount;

    std::uint32_t attempts = 0u;
    std::vector<std::uint8_t> rejectedSeeds(particleCount, 0u);
    while (minimumMembershipCount(result.membershipCounts) < minimumMemberships &&
           attempts++ < maxAttempts)
    {
        const std::uint32_t seed =
            leastCoveredParticle(result.membershipCounts, rejectedSeeds, minimumMemberships);
        if (seed >= particleCount)
        {
            break;
        }
        std::vector<std::uint32_t> members = collectConnectedGraphNeighborhood(
            restPositions, adjacencyLists, seed, targetSize, maximumSize);

        if (!inducedGraphIsConnected(adjacencyLists, members) ||
            !clusterHasSufficientExtent(restPositions, members))
        {
            rejectedSeeds[seed] = 1u;
            continue;
        }

        SoftBodyShapeMatchingCluster cluster{};
        cluster.seedParticle = seed;
        cluster.particles = std::move(members);
        cluster.blendWeights.assign(cluster.particles.size(), 1.0f);
        cluster.restCenter = weightedCenter(restPositions, {}, cluster.particles);
        for (const std::uint32_t particle : cluster.particles)
        {
            ++result.membershipCounts[particle];
        }
        result.clusters.push_back(std::move(cluster));
    }

    std::vector<float> weightSums(particleCount, 0.0f);
    for (SoftBodyShapeMatchingCluster &cluster : result.clusters)
    {
        cluster.blendWeights.resize(cluster.particles.size());
        const Diligent::float3 &seedPosition = restPositions[cluster.seedParticle];
        for (std::uint32_t i = 0u; i < static_cast<std::uint32_t>(cluster.particles.size()); ++i)
        {
            const std::uint32_t particle = cluster.particles[i];
            const float distance = std::sqrt(lengthSq(restPositions[particle] - seedPosition));
            const float rawWeight = 1.0f / (1.0e-4f + distance);
            cluster.blendWeights[i] = rawWeight;
            weightSums[particle] += rawWeight;
        }
    }

    for (SoftBodyShapeMatchingCluster &cluster : result.clusters)
    {
        for (std::uint32_t i = 0u; i < static_cast<std::uint32_t>(cluster.particles.size()); ++i)
        {
            const std::uint32_t particle = cluster.particles[i];
            if (weightSums[particle] > 0.0f)
            {
                cluster.blendWeights[i] /= weightSums[particle];
            }
        }
    }

    return result;
}

ShapeMatchingDataHost buildShapeMatchingGpuData(
    const std::vector<Diligent::float3> &restPositions,
    const std::vector<SoftBodyShapeMatchingCluster> &clusters,
    const SoftBodyShapeMatchingDesc &desc,
    const std::uint32_t globalParticleOffset,
    const std::uint32_t globalParticleCount) noexcept
{
    ShapeMatchingDataHost data{};
    data.particleMembershipRanges.resize(globalParticleCount);
    if (!desc.enabled || restPositions.empty() || clusters.empty() ||
        globalParticleOffset >= globalParticleCount)
    {
        return data;
    }

    data.solverIterations  = std::max<std::uint32_t>(1u, desc.solverIterations);
    data.maximumCorrection = std::max(desc.maximumCorrection, 0.0f);
    const std::uint32_t localParticleCapacity =
        std::min<std::uint32_t>(static_cast<std::uint32_t>(restPositions.size()),
                                globalParticleCount - globalParticleOffset);
    std::vector<std::vector<std::uint32_t>> reverseMemberships(localParticleCapacity);
    data.clusters.reserve(clusters.size());
    data.poses.reserve(clusters.size());

    for (const SoftBodyShapeMatchingCluster &cluster : clusters)
    {
        const std::uint32_t gpuClusterIndex = static_cast<std::uint32_t>(data.clusters.size());
        const std::uint32_t memberOffset = static_cast<std::uint32_t>(data.members.size());
        const std::uint32_t memberCount =
            static_cast<std::uint32_t>(std::min(cluster.particles.size(),
                                                cluster.blendWeights.size()));
        if (memberCount == 0u)
        {
            continue;
        }

        ShapeClusterGPU gpuCluster{};
        gpuCluster.memberOffset      = memberOffset;
        gpuCluster.memberCount       = memberCount;
        gpuCluster.linkOffset        = 0u;
        gpuCluster.linkCount         = 0u;
        gpuCluster.restCenterAndMass = Diligent::float4{
            cluster.restCenter.x, cluster.restCenter.y, cluster.restCenter.z,
            static_cast<float>(memberCount)};
        gpuCluster.flags      = ShapeCluster_Active;
        gpuCluster.stiffness  = desc.stiffnessPerPass;
        gpuCluster.compliance = 0.0f;

        for (std::uint32_t memberSlot = 0u; memberSlot < memberCount; ++memberSlot)
        {
            const std::uint32_t localParticle = cluster.particles[memberSlot];
            if (localParticle >= localParticleCapacity)
            {
                continue;
            }

            const Diligent::float3 restOffset = restPositions[localParticle] - cluster.restCenter;
            ShapeClusterMemberGPU member{};
            member.particleIndex  = globalParticleOffset + localParticle;
            member.fittingWeight  = 1.0f;
            member.blendWeight    = cluster.blendWeights[memberSlot];
            member.restOffset     = Diligent::float4{restOffset.x, restOffset.y, restOffset.z, 0.0f};
            reverseMemberships[localParticle].push_back(
                static_cast<std::uint32_t>(data.members.size()));
            data.members.push_back(member);
            data.membershipClusterIndices.push_back(gpuClusterIndex);
        }

        gpuCluster.memberCount = static_cast<std::uint32_t>(data.members.size()) - memberOffset;
        if (gpuCluster.memberCount == 0u)
        {
            data.members.resize(memberOffset);
            data.membershipClusterIndices.resize(memberOffset);
            continue;
        }

        ShapeClusterPoseGPU pose{};
        pose.rotationQuaternion  = Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f};
        pose.currentCenterAndStatus =
            Diligent::float4{cluster.restCenter.x, cluster.restCenter.y, cluster.restCenter.z, 1.0f};

        data.clusters.push_back(gpuCluster);
        data.poses.push_back(pose);
    }

    for (std::uint32_t localParticle = 0u; localParticle < localParticleCapacity; ++localParticle)
    {
        const std::uint32_t globalParticle = globalParticleOffset + localParticle;
        ParticleShapeMembershipRangeGPU &range = data.particleMembershipRanges[globalParticle];
        range.offset = static_cast<std::uint32_t>(data.particleMembershipIndices.size());
        range.count  = static_cast<std::uint32_t>(reverseMemberships[localParticle].size());
        data.particleMembershipIndices.insert(data.particleMembershipIndices.end(),
                                              reverseMemberships[localParticle].begin(),
                                              reverseMemberships[localParticle].end());
    }

    return data;
}

std::vector<Diligent::float3> makeShapeMatchingReferenceCube(const std::uint32_t sideCount,
                                                            const float spacing) noexcept
{
    std::vector<Diligent::float3> particles;
    const std::uint32_t clampedSide = std::max<std::uint32_t>(1u, sideCount);
    particles.reserve(clampedSide * clampedSide * clampedSide);
    const float centerOffset = 0.5f * spacing * static_cast<float>(clampedSide - 1u);
    for (std::uint32_t z = 0u; z < clampedSide; ++z)
    {
        for (std::uint32_t y = 0u; y < clampedSide; ++y)
        {
            for (std::uint32_t x = 0u; x < clampedSide; ++x)
            {
                particles.push_back({static_cast<float>(x) * spacing - centerOffset,
                                     static_cast<float>(y) * spacing - centerOffset,
                                     static_cast<float>(z) * spacing - centerOffset});
            }
        }
    }
    return particles;
}

} // namespace cressim::neo::physics
