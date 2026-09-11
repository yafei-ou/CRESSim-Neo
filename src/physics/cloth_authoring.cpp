#include "physics/cloth_authoring.h"

#include "common/math_utils_runtime.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace cressim::neo::physics
{
namespace
{
constexpr float kLengthEpsilonSq = 1.0e-12f;
constexpr float kAreaEpsilonSq   = 1.0e-16f;

bool finite(const Diligent::float3 &p)
{
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

std::uint64_t edgeKey(std::uint32_t a, std::uint32_t b)
{
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32u) | b;
}

struct TriangleKey
{
    std::array<std::uint32_t, 3> indices{};
    bool operator<(const TriangleKey &rhs) const noexcept
    {
        return indices < rhs.indices;
    }
};

struct EdgeIncidence
{
    std::uint32_t a = 0u;
    std::uint32_t b = 0u;
    std::vector<std::uint32_t> opposites;
};

float signedDihedral(const Diligent::float3 &x0, const Diligent::float3 &x1,
                     const Diligent::float3 &x2, const Diligent::float3 &x3)
{
    const Diligent::float3 edge = Diligent::normalize(x1 - x0);
    const Diligent::float3 n0   = Diligent::normalize(Diligent::cross(edge, x2 - x0));
    const Diligent::float3 n1   = Diligent::normalize(Diligent::cross(x3 - x0, edge));
    return std::atan2(Diligent::dot(edge, Diligent::cross(n0, n1)), Diligent::dot(n0, n1));
}
} // namespace

bool cookClothTopology(const ClothState &state, CookedClothTopology &outTopology,
                       std::string &errorMessage) noexcept
{
    CookedClothTopology cooked;
    errorMessage.clear();
    const auto &source = state.source;
    if (source.objectSpaceRestPositions.size() < 3u)
    {
        errorMessage = "Cloth requires at least three rest positions.";
        return false;
    }
    if (source.triangleVertexIndices.empty() || source.triangleVertexIndices.size() % 3u != 0u)
    {
        errorMessage = "Cloth requires a non-empty triangle index list divisible by three.";
        return false;
    }

    cooked.restPositions.reserve(source.objectSpaceRestPositions.size());
    for (const auto &local : source.objectSpaceRestPositions)
    {
        if (!finite(local))
        {
            errorMessage = "Cloth rest positions must be finite.";
            return false;
        }
        const auto world = common::runtime_math::applyTransform(state.restTransform, local);
        if (!finite(world))
        {
            errorMessage = "Cloth transformed rest positions must be finite.";
            return false;
        }
        cooked.restPositions.push_back(world);
    }

    std::map<std::uint64_t, EdgeIncidence> edges;
    std::set<TriangleKey> triangleKeys;
    for (std::size_t i = 0; i < source.triangleVertexIndices.size(); i += 3u)
    {
        const std::uint32_t a = source.triangleVertexIndices[i];
        const std::uint32_t b = source.triangleVertexIndices[i + 1u];
        const std::uint32_t c = source.triangleVertexIndices[i + 2u];
        if (a >= cooked.restPositions.size() || b >= cooked.restPositions.size() ||
            c >= cooked.restPositions.size())
        {
            errorMessage = "Cloth triangle index is out of range.";
            return false;
        }
        if (a == b || b == c || c == a)
        {
            errorMessage = "Cloth triangles require three distinct indices.";
            return false;
        }
        if (Diligent::dot(Diligent::cross(cooked.restPositions[b] - cooked.restPositions[a],
                                          cooked.restPositions[c] - cooked.restPositions[a]),
                          Diligent::cross(cooked.restPositions[b] - cooked.restPositions[a],
                                          cooked.restPositions[c] - cooked.restPositions[a])) <=
            kAreaEpsilonSq)
        {
            errorMessage = "Cloth triangles must have nonzero world-space area.";
            return false;
        }
        TriangleKey triangleKey{{a, b, c}};
        std::sort(triangleKey.indices.begin(), triangleKey.indices.end());
        if (!triangleKeys.insert(triangleKey).second)
        {
            errorMessage = "Cloth contains a duplicate triangle.";
            return false;
        }
        cooked.triangles.emplace_back(a, b, c);
        const std::array<std::array<std::uint32_t, 3>, 3> records{
            {{{a, b, c}}, {{b, c, a}}, {{c, a, b}}}};
        for (auto record : records)
        {
            if (record[0] > record[1]) std::swap(record[0], record[1]);
            auto &[ea, eb, opposites] = edges
                                            .try_emplace(edgeKey(record[0], record[1]),
                                                         EdgeIncidence{record[0], record[1], {}})
                                            .first->second;
            (void)ea;
            (void)eb;
            opposites.push_back(record[2]);
        }
    }

    cooked.adjacencyLists.resize(cooked.restPositions.size());
    for (auto &[key, edge] : edges)
    {
        (void)key;
        const auto delta     = cooked.restPositions[edge.b] - cooked.restPositions[edge.a];
        const float lengthSq = Diligent::dot(delta, delta);
        if (lengthSq <= kLengthEpsilonSq)
        {
            errorMessage = "Cloth generated a zero-length structural constraint.";
            return false;
        }
        cooked.structuralPairs.push_back({edge.a, edge.b});
        cooked.structuralRestLengths.push_back(std::sqrt(lengthSq));
        cooked.adjacencyLists[edge.a].push_back(edge.b);
        cooked.adjacencyLists[edge.b].push_back(edge.a);
        if (edge.opposites.size() == 2u)
        {
            std::sort(edge.opposites.begin(), edge.opposites.end());
            const auto c = edge.opposites[0];
            const auto d = edge.opposites[1];
            const float angle =
                signedDihedral(cooked.restPositions[edge.a], cooked.restPositions[edge.b],
                               cooked.restPositions[c], cooked.restPositions[d]);
            if (!std::isfinite(angle))
            {
                errorMessage = "Cloth generated a non-finite dihedral rest angle.";
                return false;
            }
            cooked.bendHinges.push_back({edge.a, edge.b, c, d, angle});
            cooked.adjacencyLists[c].push_back(d);
            cooked.adjacencyLists[d].push_back(c);
        }
    }

    cooked.staticParticleIndices = source.staticParticleIndices;
    std::sort(cooked.staticParticleIndices.begin(), cooked.staticParticleIndices.end());
    cooked.staticParticleIndices.erase(
        std::unique(cooked.staticParticleIndices.begin(), cooked.staticParticleIndices.end()),
        cooked.staticParticleIndices.end());
    if (!cooked.staticParticleIndices.empty() &&
        cooked.staticParticleIndices.back() >= cooked.restPositions.size())
    {
        errorMessage = "Cloth static particle index is out of range.";
        return false;
    }
    for (auto &neighbors : cooked.adjacencyLists)
    {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    outTopology = std::move(cooked);
    return true;
}

} // namespace cressim::neo::physics
