#ifndef CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
#define CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H

#include "common/id.h"
#include "graphics/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cressim::neo::graphics
{

struct MeshHandle
{
    common::ResourceId id = common::kInvalidResourceId;
};

struct MaterialHandle
{
    common::ResourceId id = common::kInvalidResourceId;
};

struct TextureHandle
{
    common::ResourceId id = common::kInvalidResourceId;
};

enum class MaterialProgramFamily
{
    StandardLit,
};

enum MaterialFeatureFlags : std::uint32_t
{
    MaterialFeature_None = 0u,
    MaterialFeature_AlphaTest = 1u << 0u,
    MaterialFeature_NormalMap = 1u << 1u,
    MaterialFeature_ClearCoat = 1u << 2u,
    MaterialFeature_DoubleSided = 1u << 3u,
};

constexpr std::uint32_t operator|(MaterialFeatureFlags lhs, MaterialFeatureFlags rhs) noexcept
{
    return static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs);
}

constexpr std::uint32_t operator|(std::uint32_t lhs, MaterialFeatureFlags rhs) noexcept
{
    return lhs | static_cast<std::uint32_t>(rhs);
}

constexpr bool hasMaterialFeature(std::uint32_t flags, MaterialFeatureFlags feature) noexcept
{
    return (flags & static_cast<std::uint32_t>(feature)) != 0u;
}

enum class BlendMode
{
    Opaque,
    Transparent,
};

struct MaterialPipelineDesc
{
    MaterialProgramFamily programFamily = MaterialProgramFamily::StandardLit;
    std::uint32_t featureFlags = MaterialFeature_None;
    float alphaCutoff = 0.5f;
};

struct MeshResourceDesc
{
    struct Vertex
    {
        Diligent::float3 position{};
        Diligent::float3 normal{0.0f, 1.0f, 0.0f};
        float texCoordU = 0.0f;
        float texCoordV = 0.0f;
    };

    std::string debugName;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct MaterialResourceDesc
{
    std::string debugName;
    Diligent::float3 baseColor{1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    MaterialPipelineDesc pipeline{};
    BlendMode blendMode = BlendMode::Opaque;
    float opacity = 1.0f;
    bool castsShadows = true;
    bool receivesShadows = true;
};

struct TextureResourceDesc
{
    std::string debugName;
};

class CRESSIM_NEO_GRAPHICS_API RenderResourceManager
{
public:
    MeshHandle registerMesh(const MeshResourceDesc& desc);
    MaterialHandle registerMaterial(const MaterialResourceDesc& desc);
    TextureHandle registerTexture(const TextureResourceDesc& desc);

    bool isValid(MeshHandle mesh) const;
    bool isValid(MaterialHandle material) const;
    bool isValid(TextureHandle texture) const;

    const MeshResourceDesc* tryGetMesh(MeshHandle mesh) const noexcept;
    const MaterialResourceDesc* tryGetMaterial(MaterialHandle material) const noexcept;
    const TextureResourceDesc* tryGetTexture(TextureHandle texture) const noexcept;
    bool tryGetMeshLocalBounds(MeshHandle mesh, Diligent::float3& outMin, Diligent::float3& outMax) const noexcept;

    std::uint64_t meshVersion(MeshHandle mesh) const noexcept;

private:
    struct MeshResource
    {
        MeshResourceDesc desc{};
        std::uint64_t version = 1;
        Diligent::float3 localBoundsMin{};
        Diligent::float3 localBoundsMax{};
        bool hasLocalBounds = false;
    };

    common::ResourceId mNextMeshId = 1;
    common::ResourceId mNextMaterialId = 1;
    common::ResourceId mNextTextureId = 1;

    std::unordered_map<common::ResourceId, MeshResource> mMeshes;
    std::unordered_map<common::ResourceId, MaterialResourceDesc> mMaterials;
    std::unordered_map<common::ResourceId, TextureResourceDesc> mTextures;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
