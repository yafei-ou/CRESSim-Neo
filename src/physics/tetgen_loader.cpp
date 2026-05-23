#include "physics/tetgen_loader.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace cressim::neo::physics
{

namespace
{

std::string trim(const std::string &value)
{
    const auto begin = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char ch) { return std::isspace(ch) != 0; });
    const auto end   = std::find_if_not(value.rbegin(), value.rend(),
                                        [](unsigned char ch) { return std::isspace(ch) != 0; })
                         .base();
    if (begin >= end)
    {
        return {};
    }
    return std::string(begin, end);
}

bool nextMeaningfulLine(std::ifstream &stream, std::string &line)
{
    while (std::getline(stream, line))
    {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
        {
            line.erase(comment);
        }
        line = trim(line);
        if (!line.empty())
        {
            return true;
        }
    }
    return false;
}

bool resolveTetIndicesByNodeId(const std::vector<std::int64_t> &rawIndices,
                               const std::unordered_map<std::int64_t, std::uint32_t> &idToDense,
                               std::vector<std::uint32_t> &resolved)
{
    resolved.clear();
    resolved.reserve(rawIndices.size());
    for (const std::int64_t rawIndex : rawIndices)
    {
        const auto it = idToDense.find(rawIndex);
        if (it == idToDense.end())
        {
            return false;
        }
        resolved.push_back(it->second);
    }
    return true;
}

} // namespace

bool loadTetGenFiles(const std::string &nodeFile, const std::string &eleFile, TetMeshData &outMesh,
                     std::string &errorMessage) noexcept
{
    outMesh = {};
    errorMessage.clear();

    std::ifstream nodeStream(nodeFile);
    if (!nodeStream.is_open())
    {
        errorMessage = "Unable to open TetGen node file '" + nodeFile + "'.";
        return false;
    }

    std::ifstream eleStream(eleFile);
    if (!eleStream.is_open())
    {
        errorMessage = "Unable to open TetGen ele file '" + eleFile + "'.";
        return false;
    }

    std::string line;
    if (!nextMeaningfulLine(nodeStream, line))
    {
        errorMessage = "TetGen node file '" + nodeFile + "' is empty.";
        return false;
    }

    std::istringstream nodeHeader(line);
    std::uint32_t nodeCount = 0u;
    int dimension           = 0;
    int attributes          = 0;
    int boundaryMarkers     = 0;
    if (!(nodeHeader >> nodeCount >> dimension >> attributes >> boundaryMarkers))
    {
        errorMessage = "TetGen node file '" + nodeFile + "' has an invalid header.";
        return false;
    }
    if (dimension != 3)
    {
        errorMessage = "TetGen node file '" + nodeFile + "' must describe 3D vertices.";
        return false;
    }

    outMesh.objectSpaceRestPositions.resize(nodeCount);
    std::unordered_map<std::int64_t, std::uint32_t> idToDense;
    idToDense.reserve(nodeCount);

    for (std::uint32_t i = 0u; i < nodeCount; ++i)
    {
        if (!nextMeaningfulLine(nodeStream, line))
        {
            errorMessage =
                "TetGen node file '" + nodeFile + "' ended before all declared vertices were read.";
            return false;
        }

        std::istringstream entry(line);
        std::int64_t nodeId = 0;
        Diligent::float3 position{};
        if (!(entry >> nodeId >> position.x >> position.y >> position.z))
        {
            errorMessage = "TetGen node file '" + nodeFile + "' contains an invalid vertex row.";
            return false;
        }
        if (!idToDense.emplace(nodeId, i).second)
        {
            errorMessage = "TetGen node file '" + nodeFile + "' contains duplicate node ids.";
            return false;
        }
        outMesh.objectSpaceRestPositions[i] = position;
    }

    if (!nextMeaningfulLine(eleStream, line))
    {
        errorMessage = "TetGen ele file '" + eleFile + "' is empty.";
        return false;
    }

    std::istringstream eleHeader(line);
    std::uint32_t tetCount = 0u;
    int nodesPerTet        = 0;
    int eleAttributes      = 0;
    if (!(eleHeader >> tetCount >> nodesPerTet >> eleAttributes))
    {
        errorMessage = "TetGen ele file '" + eleFile + "' has an invalid header.";
        return false;
    }
    if (nodesPerTet != 4)
    {
        errorMessage =
            "TetGen ele file '" + eleFile + "' must contain tetrahedra with exactly 4 vertices.";
        return false;
    }

    std::vector<std::int64_t> rawTetIndices;
    rawTetIndices.reserve(static_cast<std::size_t>(tetCount) * 4u);
    for (std::uint32_t i = 0u; i < tetCount; ++i)
    {
        if (!nextMeaningfulLine(eleStream, line))
        {
            errorMessage =
                "TetGen ele file '" + eleFile + "' ended before all declared tetrahedra were read.";
            return false;
        }

        std::istringstream entry(line);
        std::int64_t elementId = 0;
        std::int64_t i0        = 0;
        std::int64_t i1        = 0;
        std::int64_t i2        = 0;
        std::int64_t i3        = 0;
        if (!(entry >> elementId >> i0 >> i1 >> i2 >> i3))
        {
            errorMessage = "TetGen ele file '" + eleFile + "' contains an invalid tetrahedron row.";
            return false;
        }
        rawTetIndices.push_back(i0);
        rawTetIndices.push_back(i1);
        rawTetIndices.push_back(i2);
        rawTetIndices.push_back(i3);
    }

    // Accept only TetGen-style element connectivity that references node ids from the
    // corresponding .node table. Dense 0-based or 1-based reindexings from third-party exporters
    // are rejected here on purpose so we never silently load an ambiguous topology.
    std::vector<std::uint32_t> byNodeId;
    if (resolveTetIndicesByNodeId(rawTetIndices, idToDense, byNodeId))
    {
        outMesh.tetVertexIndices = std::move(byNodeId);
        return true;
    }

    errorMessage =
        "TetGen ele file '" + eleFile +
        "' must reference node ids from the matching TetGen node table; alternate dense index "
        "reindexings are not supported.";
    return false;
}

} // namespace cressim::neo::physics
