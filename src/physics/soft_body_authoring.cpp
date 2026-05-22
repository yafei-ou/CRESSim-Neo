#include "physics/soft_body_authoring.h"
#include "common/math_utils_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace cressim::neo::physics
{

namespace
{

constexpr float kMinExtent     = 1.0e-4f;
constexpr float kMinSpacing    = 1.0e-4f;
constexpr float kMinRestVolume = 1.0e-8f;

std::uint32_t flattenGridIndex(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                               const Diligent::uint3 &resolution)
{
    return x * resolution.y * resolution.z + y * resolution.z + z;
}

float signedTetVolume(const Diligent::float3 &a, const Diligent::float3 &b,
                      const Diligent::float3 &c, const Diligent::float3 &d)
{
    return Diligent::dot(Diligent::cross(b - a, c - a), d - a) / 6.0f;
}

std::uint64_t sortedEdgeKey(const std::uint32_t a, const std::uint32_t b)
{
    const std::uint32_t lo = std::min(a, b);
    const std::uint32_t hi = std::max(a, b);
    return (static_cast<std::uint64_t>(lo) << 32u) | hi;
}

std::uint64_t sortedFaceKey(std::uint32_t a, std::uint32_t b, std::uint32_t c)
{
    if (a > b)
    {
        std::swap(a, b);
    }
    if (b > c)
    {
        std::swap(b, c);
    }
    if (a > b)
    {
        std::swap(a, b);
    }
    return (static_cast<std::uint64_t>(a) << 42u) | (static_cast<std::uint64_t>(b) << 21u) |
           static_cast<std::uint64_t>(c);
}

std::vector<std::uint32_t> normalizedStaticParticleIndices(
    const std::vector<std::uint32_t> &indices, const std::uint32_t particleCount,
    std::string &errorMessage)
{
    std::vector<std::uint32_t> result = indices;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    for (const std::uint32_t index : result)
    {
        if (index >= particleCount)
        {
            std::ostringstream stream;
            stream << "Static particle index " << index << " is out of range for " << particleCount
                   << " particles.";
            errorMessage = stream.str();
            return {};
        }
    }
    return result;
}

bool buildRegularGridSource(const SoftBodyRegularGridSource &source, TetMeshData &mesh,
                            std::string &errorMessage)
{
    mesh = {};

    const Diligent::float3 size{
        std::max(source.size.x, kMinExtent),
        std::max(source.size.y, kMinExtent),
        std::max(source.size.z, kMinExtent),
    };
    const float spacing = std::max(source.targetParticleSpacing, kMinSpacing);

    const auto deriveResolution = [spacing](const float extent) -> std::uint32_t
    {
        return std::max<std::uint32_t>(2u, static_cast<std::uint32_t>(std::ceil(extent / spacing)) +
                                               1u);
    };

    const Diligent::uint3 resolution{
        deriveResolution(size.x),
        deriveResolution(size.y),
        deriveResolution(size.z),
    };

    mesh.objectSpaceRestPositions.reserve(static_cast<std::size_t>(resolution.x) * resolution.y *
                                          resolution.z);

    for (std::uint32_t x = 0u; x < resolution.x; ++x)
    {
        const float tx = resolution.x > 1u
                             ? static_cast<float>(x) / static_cast<float>(resolution.x - 1u)
                             : 0.0f;
        for (std::uint32_t y = 0u; y < resolution.y; ++y)
        {
            const float ty = resolution.y > 1u
                                 ? static_cast<float>(y) / static_cast<float>(resolution.y - 1u)
                                 : 0.0f;
            for (std::uint32_t z = 0u; z < resolution.z; ++z)
            {
                const float tz = resolution.z > 1u
                                     ? static_cast<float>(z) / static_cast<float>(resolution.z - 1u)
                                     : 0.0f;
                mesh.objectSpaceRestPositions.push_back(Diligent::float3{
                    (tx - 0.5f) * size.x, (ty - 0.5f) * size.y, (tz - 0.5f) * size.z});
            }
        }
    }

    mesh.tetVertexIndices.reserve(static_cast<std::size_t>(resolution.x - 1u) *
                                  (resolution.y - 1u) * (resolution.z - 1u) * 20u);
    for (std::uint32_t x = 0u; x + 1u < resolution.x; ++x)
    {
        for (std::uint32_t y = 0u; y + 1u < resolution.y; ++y)
        {
            for (std::uint32_t z = 0u; z + 1u < resolution.z; ++z)
            {
                const std::uint32_t p0 = flattenGridIndex(x, y, z, resolution);
                const std::uint32_t p1 = flattenGridIndex(x, y, z + 1u, resolution);
                const std::uint32_t p3 = flattenGridIndex(x + 1u, y, z, resolution);
                const std::uint32_t p2 = flattenGridIndex(x + 1u, y, z + 1u, resolution);
                const std::uint32_t p7 = flattenGridIndex(x + 1u, y + 1u, z, resolution);
                const std::uint32_t p6 = flattenGridIndex(x + 1u, y + 1u, z + 1u, resolution);
                const std::uint32_t p4 = flattenGridIndex(x, y + 1u, z, resolution);
                const std::uint32_t p5 = flattenGridIndex(x, y + 1u, z + 1u, resolution);

                const std::array<std::array<std::uint32_t, 4>, 5> oddSplit  = {{
                    {{p2, p1, p6, p3}},
                    {{p6, p3, p4, p7}},
                    {{p4, p1, p6, p5}},
                    {{p3, p1, p4, p0}},
                    {{p6, p1, p4, p3}},
                }};
                const std::array<std::array<std::uint32_t, 4>, 5> evenSplit = {{
                    {{p0, p2, p5, p1}},
                    {{p7, p2, p0, p3}},
                    {{p5, p2, p7, p6}},
                    {{p7, p0, p5, p4}},
                    {{p0, p2, p7, p5}},
                }};

                const auto &split = (((x + y + z) & 1u) != 0u) ? oddSplit : evenSplit;
                for (const auto &tet : split)
                {
                    mesh.tetVertexIndices.insert(mesh.tetVertexIndices.end(), tet.begin(),
                                                 tet.end());
                }
            }
        }
    }

    if (mesh.tetVertexIndices.empty())
    {
        errorMessage = "RegularGrid source did not generate any tetrahedra.";
        return false;
    }
    return true;
}

void buildMeshfreeKNearestGraph(const std::vector<Diligent::float3> &restPositions,
                                std::uint32_t neighbourCount,
                                ResolvedSoftBodyTopology &outTopology)
{
    const std::uint32_t particleCount = static_cast<std::uint32_t>(restPositions.size());
    if (particleCount < 2u)
    {
        return;
    }

    neighbourCount = std::min(neighbourCount, particleCount - 1u);
    std::set<std::uint64_t> uniqueEdges;
    std::vector<std::pair<float, std::uint32_t>> nearest;
    nearest.reserve(particleCount - 1u);

    for (std::uint32_t particleIndex = 0u; particleIndex < particleCount; ++particleIndex)
    {
        nearest.clear();
        const Diligent::float3 &position = restPositions[particleIndex];
        for (std::uint32_t candidateIndex = 0u; candidateIndex < particleCount; ++candidateIndex)
        {
            if (candidateIndex == particleIndex)
            {
                continue;
            }

            const Diligent::float3 delta = restPositions[candidateIndex] - position;
            nearest.emplace_back(Diligent::dot(delta, delta), candidateIndex);
        }

        std::sort(nearest.begin(), nearest.end(),
            [](const auto &lhs, const auto &rhs)
            {
                if (lhs.first != rhs.first)
                {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            }
        );

        for (std::uint32_t neighbourSlot = 0u; neighbourSlot < neighbourCount; ++neighbourSlot)
        {
            const std::uint32_t neighbourIndex = nearest[neighbourSlot].second;
            outTopology.adjacencyLists[particleIndex].push_back(neighbourIndex);
            outTopology.adjacencyLists[neighbourIndex].push_back(particleIndex);

            if (uniqueEdges.insert(sortedEdgeKey(particleIndex, neighbourIndex)).second)
            {
                outTopology.edges.push_back({std::min(particleIndex, neighbourIndex),
                                             std::max(particleIndex, neighbourIndex)});
            }
        }
    }
}

} // namespace

bool resolveSoftBodyTopology(const SoftBodyState &state, const TetMeshData *tetGenCache,
                             ResolvedSoftBodyTopology &outTopology,
                             std::string &errorMessage) noexcept
{
    outTopology = {};
    errorMessage.clear();

    TetMeshData sourceMesh;
    const std::vector<std::uint32_t> *staticParticleIndices = nullptr;
    switch (state.source.kind)
    {
    case SoftBodySourceKind::RegularGrid:
        if (!buildRegularGridSource(state.source.regularGrid, sourceMesh, errorMessage))
        {
            return false;
        }
        staticParticleIndices = &state.source.regularGrid.staticParticleIndices;
        break;
    case SoftBodySourceKind::TetMesh:
        sourceMesh.objectSpaceRestPositions = state.source.tetMesh.objectSpaceRestPositions;
        sourceMesh.tetVertexIndices         = state.source.tetMesh.tetVertexIndices;
        staticParticleIndices               = &state.source.tetMesh.staticParticleIndices;
        break;
    case SoftBodySourceKind::TetGenFiles:
        if (tetGenCache == nullptr)
        {
            errorMessage = "TetGenFiles source is missing its resolved mesh cache.";
            return false;
        }
        sourceMesh.objectSpaceRestPositions = tetGenCache->objectSpaceRestPositions;
        sourceMesh.tetVertexIndices         = tetGenCache->tetVertexIndices;
        staticParticleIndices               = &state.source.tetGen.staticParticleIndices;
        break;
    case SoftBodySourceKind::MeshfreeParticles:
        sourceMesh.objectSpaceRestPositions =
            state.source.meshfreeParticles.particleRestPositions;
        staticParticleIndices = &state.source.meshfreeParticles.staticParticleIndices;
        break;
    }

    if (sourceMesh.objectSpaceRestPositions.empty())
    {
        errorMessage = "Soft body source does not contain any rest positions.";
        return false;
    }
    const bool meshfreeSource = state.source.kind == SoftBodySourceKind::MeshfreeParticles;
    if (!meshfreeSource && sourceMesh.tetVertexIndices.empty())
    {
        errorMessage = "Soft body source does not contain any tetrahedra.";
        return false;
    }
    if (!meshfreeSource && (sourceMesh.tetVertexIndices.size() % 4u) != 0u)
    {
        errorMessage = "Soft body tetrahedron index buffer size must be divisible by 4.";
        return false;
    }

    outTopology.restPositions.reserve(sourceMesh.objectSpaceRestPositions.size());
    for (const Diligent::float3 &objectSpacePosition : sourceMesh.objectSpaceRestPositions)
    {
        outTopology.restPositions.push_back(
            common::runtime_math::applyTransform(state.restTransform, objectSpacePosition));
    }

    const std::uint32_t particleCount =
        static_cast<std::uint32_t>(outTopology.restPositions.size());
    outTopology.adjacencyLists.resize(particleCount);

    if (meshfreeSource)
    {
        buildMeshfreeKNearestGraph(outTopology.restPositions,
                                   state.source.meshfreeParticles.neighbourCount, outTopology);

        if (staticParticleIndices != nullptr)
        {
            outTopology.staticParticleIndices =
                normalizedStaticParticleIndices(*staticParticleIndices, particleCount, errorMessage);
            if (!errorMessage.empty())
            {
                return false;
            }
        }
        return true;
    }

    std::set<std::uint64_t> uniqueEdges;
    struct BoundaryFaceEntry
    {
        Diligent::uint3 face{0u, 0u, 0u};
        std::uint32_t count = 0u;
    };
    std::unordered_map<std::uint64_t, BoundaryFaceEntry> faceCounts;
    outTopology.tets.reserve(sourceMesh.tetVertexIndices.size() / 4u);
    for (std::size_t tetBase = 0u; tetBase < sourceMesh.tetVertexIndices.size(); tetBase += 4u)
    {
        const std::array<std::uint32_t, 4> tet = {
            sourceMesh.tetVertexIndices[tetBase + 0u],
            sourceMesh.tetVertexIndices[tetBase + 1u],
            sourceMesh.tetVertexIndices[tetBase + 2u],
            sourceMesh.tetVertexIndices[tetBase + 3u],
        };

        for (const std::uint32_t index : tet)
        {
            if (index >= particleCount)
            {
                std::ostringstream stream;
                stream << "Tet vertex index " << index << " is out of range for " << particleCount
                       << " particles.";
                errorMessage = stream.str();
                return false;
            }
        }

        if (tet[0] == tet[1] || tet[0] == tet[2] || tet[0] == tet[3] || tet[1] == tet[2] ||
            tet[1] == tet[3] || tet[2] == tet[3])
        {
            errorMessage = "Soft body tetrahedra must not repeat a vertex index.";
            return false;
        }

        const float restVolume = std::abs(
            signedTetVolume(outTopology.restPositions[tet[0]], outTopology.restPositions[tet[1]],
                            outTopology.restPositions[tet[2]], outTopology.restPositions[tet[3]]));
        if (restVolume <= kMinRestVolume)
        {
            errorMessage = "Soft body source contains a degenerate tetrahedron.";
            return false;
        }

        outTopology.tets.push_back(tet);

        const std::array<Diligent::uint3, 4> tetFaces = {{
            Diligent::uint3{tet[0], tet[2], tet[1]},
            Diligent::uint3{tet[0], tet[1], tet[3]},
            Diligent::uint3{tet[0], tet[3], tet[2]},
            Diligent::uint3{tet[1], tet[2], tet[3]},
        }};
        for (const Diligent::uint3 &face : tetFaces)
        {
            const std::uint64_t key  = sortedFaceKey(face.x, face.y, face.z);
            BoundaryFaceEntry &entry = faceCounts[key];
            if (entry.count == 0u)
            {
                entry.face = face;
            }
            ++entry.count;
        }

        const std::array<std::array<std::uint32_t, 2>, 6> tetEdges = {{
            {{tet[0], tet[1]}},
            {{tet[0], tet[2]}},
            {{tet[0], tet[3]}},
            {{tet[1], tet[2]}},
            {{tet[1], tet[3]}},
            {{tet[2], tet[3]}},
        }};

        for (const auto &edge : tetEdges)
        {
            outTopology.adjacencyLists[edge[0]].push_back(edge[1]);
            outTopology.adjacencyLists[edge[1]].push_back(edge[0]);
            if (uniqueEdges.insert(sortedEdgeKey(edge[0], edge[1])).second)
            {
                outTopology.edges.push_back(
                    {std::min(edge[0], edge[1]), std::max(edge[0], edge[1])});
            }
        }
    }

    for (auto &neighbors : outTopology.adjacencyLists)
    {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    outTopology.boundaryFaces.clear();
    outTopology.boundaryFaces.reserve(faceCounts.size());
    for (const auto &[key, entry] : faceCounts)
    {
        (void)key;
        if (entry.count == 1u)
        {
            outTopology.boundaryFaces.push_back(entry.face);
        }
    }

    if (staticParticleIndices != nullptr)
    {
        outTopology.staticParticleIndices =
            normalizedStaticParticleIndices(*staticParticleIndices, particleCount, errorMessage);
        if (!errorMessage.empty())
        {
            return false;
        }
    }

    return true;
}

} // namespace cressim::neo::physics
