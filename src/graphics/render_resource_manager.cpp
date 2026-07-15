#include "graphics/render_resource_manager.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace cressim::neo::graphics
{

namespace
{

constexpr float kTangentEpsilon = 1.0e-6f;

std::uint32_t arrayLayerCount(TextureDimension dimension)
{
    return dimension == TextureDimension::TextureCube ? 6u : 1u;
}

Diligent::float3 normalizeOrFallback(const Diligent::float3 &value,
                                     const Diligent::float3 &fallback)
{
    const float lenSq = Diligent::dot(value, value);
    if (lenSq <= kTangentEpsilon)
    {
        return fallback;
    }

    const float invLen = 1.0f / std::sqrt(lenSq);
    return value * invLen;
}

Diligent::float3 buildStableTangent(const Diligent::float3 &normal)
{
    const Diligent::float3 safeNormal =
        normalizeOrFallback(normal, Diligent::float3{0.0f, 1.0f, 0.0f});
    const Diligent::float3 referenceAxis = std::fabs(safeNormal.y) < 0.999f
                                               ? Diligent::float3{0.0f, 1.0f, 0.0f}
                                               : Diligent::float3{1.0f, 0.0f, 0.0f};
    return normalizeOrFallback(Diligent::cross(referenceAxis, safeNormal),
                               Diligent::float3{1.0f, 0.0f, 0.0f});
}

bool hasValidTangents(const MeshResourceDesc &desc)
{
    for (const MeshResourceDesc::Vertex &vertex : desc.vertices)
    {
        const Diligent::float3 tangent{vertex.tangent.x, vertex.tangent.y, vertex.tangent.z};
        if (Diligent::dot(tangent, tangent) <= kTangentEpsilon ||
            std::fabs(vertex.tangent.w) <= kTangentEpsilon)
        {
            return false;
        }
    }
    return !desc.vertices.empty();
}

void generateTangents(MeshResourceDesc &desc)
{
    if (desc.vertices.empty())
    {
        return;
    }

    if (hasValidTangents(desc))
    {
        for (MeshResourceDesc::Vertex &vertex : desc.vertices)
        {
            const Diligent::float3 normal =
                normalizeOrFallback(vertex.normal, Diligent::float3{0.0f, 1.0f, 0.0f});
            Diligent::float3 tangent{vertex.tangent.x, vertex.tangent.y, vertex.tangent.z};
            tangent        = tangent - normal * Diligent::dot(normal, tangent);
            tangent        = normalizeOrFallback(tangent, buildStableTangent(normal));
            vertex.normal  = normal;
            vertex.tangent = {tangent.x, tangent.y, tangent.z,
                              vertex.tangent.w >= 0.0f ? 1.0f : -1.0f};
        }
        return;
    }

    std::vector<Diligent::float3> tangentSum(desc.vertices.size(),
                                             Diligent::float3{0.0f, 0.0f, 0.0f});
    std::vector<Diligent::float3> bitangentSum(desc.vertices.size(),
                                               Diligent::float3{0.0f, 0.0f, 0.0f});

    const std::size_t triangleCount = desc.indices.size() / 3u;
    for (std::size_t triangleIdx = 0; triangleIdx < triangleCount; ++triangleIdx)
    {
        const std::uint32_t i0 = desc.indices[triangleIdx * 3u + 0u];
        const std::uint32_t i1 = desc.indices[triangleIdx * 3u + 1u];
        const std::uint32_t i2 = desc.indices[triangleIdx * 3u + 2u];
        if (i0 >= desc.vertices.size() || i1 >= desc.vertices.size() || i2 >= desc.vertices.size())
        {
            continue;
        }

        const MeshResourceDesc::Vertex &v0 = desc.vertices[i0];
        const MeshResourceDesc::Vertex &v1 = desc.vertices[i1];
        const MeshResourceDesc::Vertex &v2 = desc.vertices[i2];

        const Diligent::float3 edge1 = v1.position - v0.position;
        const Diligent::float3 edge2 = v2.position - v0.position;
        const float du1              = v1.texCoordU - v0.texCoordU;
        const float dv1              = v1.texCoordV - v0.texCoordV;
        const float du2              = v2.texCoordU - v0.texCoordU;
        const float dv2              = v2.texCoordV - v0.texCoordV;
        const float denom            = du1 * dv2 - du2 * dv1;
        if (std::fabs(denom) <= kTangentEpsilon)
        {
            continue;
        }

        const float invDenom                     = 1.0f / denom;
        const Diligent::float3 triangleTangent   = (edge1 * dv2 - edge2 * dv1) * invDenom;
        const Diligent::float3 triangleBitangent = (edge2 * du1 - edge1 * du2) * invDenom;

        tangentSum[i0] += triangleTangent;
        tangentSum[i1] += triangleTangent;
        tangentSum[i2] += triangleTangent;
        bitangentSum[i0] += triangleBitangent;
        bitangentSum[i1] += triangleBitangent;
        bitangentSum[i2] += triangleBitangent;
    }

    for (std::size_t vertexIdx = 0; vertexIdx < desc.vertices.size(); ++vertexIdx)
    {
        MeshResourceDesc::Vertex &vertex = desc.vertices[vertexIdx];
        const Diligent::float3 normal =
            normalizeOrFallback(vertex.normal, Diligent::float3{0.0f, 1.0f, 0.0f});
        Diligent::float3 tangent =
            tangentSum[vertexIdx] - normal * Diligent::dot(normal, tangentSum[vertexIdx]);
        tangent = normalizeOrFallback(tangent, buildStableTangent(normal));

        Diligent::float3 bitangent = bitangentSum[vertexIdx];
        bitangent                  = bitangent - normal * Diligent::dot(normal, bitangent);
        const Diligent::float3 derivedBitangent = Diligent::cross(normal, tangent);
        const float handedness = Diligent::dot(derivedBitangent, bitangent) < 0.0f ? -1.0f : 1.0f;

        vertex.normal  = normal;
        vertex.tangent = {tangent.x, tangent.y, tangent.z, handedness};
    }
}

MaterialResourceDesc normalizeMaterialDesc(const MaterialResourceDesc &desc)
{
    MaterialResourceDesc normalized  = desc;
    normalized.pipeline.featureFlags = effectiveMaterialFeatureFlags(normalized);
    if (normalized.normalTexture.id != common::kInvalidResourceId)
    {
        normalized.pipeline.featureFlags |= MaterialFeatureFlags::NormalMap;
    }
    return normalized;
}

TextureResourceDesc normalizeTextureDesc(const TextureResourceDesc &desc)
{
    TextureResourceDesc normalized = desc;
    normalized.width               = std::max(normalized.width, 1u);
    normalized.height              = std::max(normalized.height, 1u);
    normalized.mipLevelCount       = std::max(normalized.mipLevelCount, 1u);

    const std::uint32_t expectedSubresourceCount =
        normalized.mipLevelCount * arrayLayerCount(normalized.dimension);
    if (normalized.subresources.empty() && !normalized.pixelData.empty())
    {
        normalized.subresources.resize(expectedSubresourceCount);
        normalized.subresources.front().pixelData = normalized.pixelData;
    }
    return normalized;
}

} // namespace

struct RenderResourceManager::Impl
{
    struct MeshResource
    {
        MeshResourceDesc desc{};
        std::uint64_t version = 1;
        Diligent::float3 localBoundsMin{};
        Diligent::float3 localBoundsMax{};
        bool hasLocalBounds = false;
    };

    common::ResourceId mNextMeshId     = 1;
    common::ResourceId mNextMaterialId = 1;
    common::ResourceId mNextTextureId  = 1;
    std::unordered_map<common::ResourceId, MeshResource> mMeshes;
    std::unordered_map<common::ResourceId, MaterialResourceDesc> mMaterials;
    std::unordered_map<common::ResourceId, TextureResourceDesc> mTextures;
};

RenderResourceManager::RenderResourceManager() : mImpl(std::make_unique<Impl>()) {}

RenderResourceManager::~RenderResourceManager() = default;

RenderResourceManager::RenderResourceManager(const RenderResourceManager &other)
    : mImpl(std::make_unique<Impl>(*other.mImpl))
{
}

RenderResourceManager &RenderResourceManager::operator=(const RenderResourceManager &other)
{
    if (this != &other)
    {
        mImpl = std::make_unique<Impl>(*other.mImpl);
    }
    return *this;
}

RenderResourceManager::RenderResourceManager(RenderResourceManager &&other) noexcept = default;

RenderResourceManager &RenderResourceManager::operator=(RenderResourceManager &&other) noexcept =
    default;

MeshHandle RenderResourceManager::registerMesh(const MeshResourceDesc &desc)
{
    const common::ResourceId id = mImpl->mNextMeshId++;
    Impl::MeshResource resource{};
    resource.desc = desc;
    generateTangents(resource.desc);
    if (!desc.vertices.empty())
    {
        resource.localBoundsMin = resource.desc.vertices.front().position;
        resource.localBoundsMax = resource.desc.vertices.front().position;
        for (const MeshResourceDesc::Vertex &vertex : resource.desc.vertices)
        {
            resource.localBoundsMin.x = std::min(resource.localBoundsMin.x, vertex.position.x);
            resource.localBoundsMin.y = std::min(resource.localBoundsMin.y, vertex.position.y);
            resource.localBoundsMin.z = std::min(resource.localBoundsMin.z, vertex.position.z);
            resource.localBoundsMax.x = std::max(resource.localBoundsMax.x, vertex.position.x);
            resource.localBoundsMax.y = std::max(resource.localBoundsMax.y, vertex.position.y);
            resource.localBoundsMax.z = std::max(resource.localBoundsMax.z, vertex.position.z);
        }
        resource.hasLocalBounds = true;
    }
    mImpl->mMeshes.emplace(id, std::move(resource));
    return MeshHandle{id};
}

MaterialHandle RenderResourceManager::registerMaterial(const MaterialResourceDesc &desc)
{
    const common::ResourceId id = mImpl->mNextMaterialId++;
    mImpl->mMaterials.emplace(id, normalizeMaterialDesc(desc));
    return MaterialHandle{id};
}

TextureHandle RenderResourceManager::registerTexture(const TextureResourceDesc &desc)
{
    const common::ResourceId id = mImpl->mNextTextureId++;
    mImpl->mTextures.emplace(id, normalizeTextureDesc(desc));
    return TextureHandle{id};
}

bool RenderResourceManager::isValid(MeshHandle mesh) const
{
    return mImpl->mMeshes.find(mesh.id) != mImpl->mMeshes.end();
}

bool RenderResourceManager::isValid(MaterialHandle material) const
{
    return mImpl->mMaterials.find(material.id) != mImpl->mMaterials.end();
}

bool RenderResourceManager::isValid(TextureHandle texture) const
{
    return mImpl->mTextures.find(texture.id) != mImpl->mTextures.end();
}

const MeshResourceDesc *RenderResourceManager::tryGetMesh(MeshHandle mesh) const noexcept
{
    const auto it = mImpl->mMeshes.find(mesh.id);
    if (it == mImpl->mMeshes.end())
    {
        return nullptr;
    }
    return &it->second.desc;
}

const MaterialResourceDesc *RenderResourceManager::tryGetMaterial(
    MaterialHandle material) const noexcept
{
    const auto it = mImpl->mMaterials.find(material.id);
    if (it == mImpl->mMaterials.end())
    {
        return nullptr;
    }
    return &it->second;
}

const TextureResourceDesc *RenderResourceManager::tryGetTexture(
    TextureHandle texture) const noexcept
{
    const auto it = mImpl->mTextures.find(texture.id);
    if (it == mImpl->mTextures.end())
    {
        return nullptr;
    }
    return &it->second;
}

bool RenderResourceManager::tryGetMeshLocalBounds(MeshHandle mesh, Diligent::float3 &outMin,
                                                  Diligent::float3 &outMax) const noexcept
{
    const auto it = mImpl->mMeshes.find(mesh.id);
    if (it == mImpl->mMeshes.end() || !it->second.hasLocalBounds)
    {
        outMin = {};
        outMax = {};
        return false;
    }

    outMin = it->second.localBoundsMin;
    outMax = it->second.localBoundsMax;
    return true;
}

std::uint64_t RenderResourceManager::meshVersion(MeshHandle mesh) const noexcept
{
    const auto it = mImpl->mMeshes.find(mesh.id);
    if (it == mImpl->mMeshes.end())
    {
        return 0;
    }
    return it->second.version;
}

} // namespace cressim::neo::graphics
