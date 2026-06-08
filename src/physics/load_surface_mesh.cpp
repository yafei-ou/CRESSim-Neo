#include "physics/load_surface_mesh.h"

#include "common/logger.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cressim::neo::physics
{

namespace
{

std::int32_t resolveObjIndex(std::int32_t index, std::size_t count) noexcept
{
    if (index > 0)
    {
        return index - 1;
    }
    if (index < 0)
    {
        return static_cast<std::int32_t>(count) + index;
    }
    return -1;
}

bool parseObjFaceVertex(const std::string &token, std::int32_t &positionIndex,
                        std::int32_t &normalIndex)
{
    positionIndex = -1;
    normalIndex   = -1;

    const std::size_t firstSlash = token.find('/');
    if (firstSlash == std::string::npos)
    {
        positionIndex = std::stoi(token);
        return true;
    }

    positionIndex = std::stoi(token.substr(0u, firstSlash));
    const std::size_t secondSlash = token.find('/', firstSlash + 1u);
    if (secondSlash != std::string::npos && secondSlash + 1u < token.size())
    {
        normalIndex = std::stoi(token.substr(secondSlash + 1u));
    }
    return true;
}

void recomputeNormals(SurfaceMeshData &mesh)
{
    mesh.surfaceNormals.assign(mesh.surfaceRestPositions.size(),
                               Diligent::float3{0.0f, 0.0f, 0.0f});

    for (const Diligent::uint3 &triangle : mesh.surfaceTriangles)
    {
        if (triangle.x >= mesh.surfaceRestPositions.size() ||
            triangle.y >= mesh.surfaceRestPositions.size() ||
            triangle.z >= mesh.surfaceRestPositions.size())
        {
            continue;
        }

        const Diligent::float3 &p0 = mesh.surfaceRestPositions[triangle.x];
        const Diligent::float3 &p1 = mesh.surfaceRestPositions[triangle.y];
        const Diligent::float3 &p2 = mesh.surfaceRestPositions[triangle.z];
        const Diligent::float3 faceNormal = Diligent::cross(p2 - p0, p1 - p0);
        if (Diligent::dot(faceNormal, faceNormal) <= 1.0e-12f)
        {
            continue;
        }

        const Diligent::float3 normalizedFaceNormal = Diligent::normalize(faceNormal);
        mesh.surfaceNormals[triangle.x] += normalizedFaceNormal;
        mesh.surfaceNormals[triangle.y] += normalizedFaceNormal;
        mesh.surfaceNormals[triangle.z] += normalizedFaceNormal;
    }

    for (Diligent::float3 &normal : mesh.surfaceNormals)
    {
        if (Diligent::dot(normal, normal) > 1.0e-8f)
        {
            normal = Diligent::normalize(normal);
        }
        else
        {
            normal = {0.0f, 1.0f, 0.0f};
        }
    }
}

} // namespace

bool readObjSurfaceMesh(const std::filesystem::path &path, SurfaceMeshData &mesh,
                        std::string &errorMessage)
{
    mesh = {};
    errorMessage.clear();

    std::ifstream stream(path);
    if (!stream)
    {
        errorMessage = "Failed to open OBJ surface mesh: " + path.string();
        return false;
    }

    std::vector<Diligent::float3> positions;
    std::vector<Diligent::float3> normals;
    std::unordered_map<std::string, std::uint32_t> vertexCache;
    std::string line;
    try
    {
        while (std::getline(stream, line))
        {
            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            std::istringstream lineStream(line);
            std::string keyword;
            lineStream >> keyword;
            if (keyword == "v")
            {
                Diligent::float3 position{};
                lineStream >> position.x >> position.y >> position.z;
                positions.push_back(position);
            }
            else if (keyword == "vn")
            {
                Diligent::float3 normal{};
                lineStream >> normal.x >> normal.y >> normal.z;
                normals.push_back(normal);
            }
            else if (keyword == "f")
            {
                std::vector<std::string> faceVertices;
                std::string token;
                while (lineStream >> token)
                {
                    faceVertices.push_back(token);
                }
                if (faceVertices.size() < 3u)
                {
                    continue;
                }

                const auto emitVertex = [&](const std::string &faceToken) -> std::uint32_t {
                    if (const auto it = vertexCache.find(faceToken); it != vertexCache.end())
                    {
                        return it->second;
                    }

                    std::int32_t positionIndex = -1;
                    std::int32_t normalIndex   = -1;
                    parseObjFaceVertex(faceToken, positionIndex, normalIndex);
                    positionIndex = resolveObjIndex(positionIndex, positions.size());
                    normalIndex   = resolveObjIndex(normalIndex, normals.size());
                    if (positionIndex < 0 ||
                        static_cast<std::size_t>(positionIndex) >= positions.size())
                    {
                        throw std::runtime_error("OBJ face references an invalid position index.");
                    }

                    const std::uint32_t vertexIndex =
                        static_cast<std::uint32_t>(mesh.surfaceRestPositions.size());
                    mesh.surfaceRestPositions.push_back(
                        positions[static_cast<std::size_t>(positionIndex)]);
                    if (normalIndex >= 0 && static_cast<std::size_t>(normalIndex) < normals.size())
                    {
                        mesh.surfaceNormals.push_back(
                            normals[static_cast<std::size_t>(normalIndex)]);
                    }
                    else
                    {
                        mesh.surfaceNormals.push_back({0.0f, 0.0f, 0.0f});
                    }
                    vertexCache.emplace(faceToken, vertexIndex);
                    return vertexIndex;
                };

                const std::uint32_t i0 = emitVertex(faceVertices[0]);
                for (std::size_t i = 1u; i + 1u < faceVertices.size(); ++i)
                {
                    const std::uint32_t i1 = emitVertex(faceVertices[i]);
                    const std::uint32_t i2 = emitVertex(faceVertices[i + 1u]);
                    mesh.surfaceTriangles.emplace_back(i0, i2, i1);
                }
            }
        }
    }
    catch (const std::exception &error)
    {
        errorMessage = "Failed to parse OBJ surface mesh " + path.string() + ": " + error.what();
        mesh         = {};
        return false;
    }

    const bool hasValidImportedNormals =
        mesh.surfaceNormals.size() == mesh.surfaceRestPositions.size() &&
        std::all_of(mesh.surfaceNormals.begin(), mesh.surfaceNormals.end(),
                    [](const Diligent::float3 &normal) {
                        return Diligent::dot(normal, normal) > 1.0e-8f;
                    });
    if (!hasValidImportedNormals)
    {
        recomputeNormals(mesh);
    }

    if (mesh.surfaceRestPositions.empty() || mesh.surfaceTriangles.empty())
    {
        errorMessage = "OBJ surface mesh is empty or invalid: " + path.string();
        mesh         = {};
        return false;
    }

    return true;
}

SurfaceMeshData loadObjSurfaceMesh(const std::filesystem::path &path)
{
    SurfaceMeshData mesh;
    std::string errorMessage;
    if (!readObjSurfaceMesh(path, mesh, errorMessage))
    {
        CRESSIM_LOG_ERROR(errorMessage);
        return {};
    }
    return mesh;
}

} // namespace cressim::neo::physics
