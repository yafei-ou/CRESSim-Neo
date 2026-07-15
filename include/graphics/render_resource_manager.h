#ifndef CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
#define CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H

#include "common/flags.h"
#include "common/id.h"
#include "graphics/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>
#include <memory>
#include <string>
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
    SoftBodyLit,
    CurveLit,
};

enum class MaterialRenderMode
{
    Opaque,
    Cutout,
    Transparent,
};

enum class TextureColorSpace
{
    Linear,
    Srgb,
};

enum class TexturePixelFormat
{
    RGBA8,
    RGBA16F,
};

enum class TextureMipPolicy
{
    Disabled,
    Generate,
};

enum class TextureDimension
{
    Texture2D,
    TextureCube,
};

enum class IblQualityTier : std::uint32_t
{
    Off         = 0u,
    DiffuseOnly = 1u,
    Full        = 2u,
};

enum class MaterialFeatureFlags : std::uint32_t
{
    None        = 0u,
    AlphaTest   = 1u << 0u,
    NormalMap   = 1u << 1u,
    ClearCoat   = 1u << 2u,
    DoubleSided = 1u << 3u,
};
CRESSIM_NEO_DEFINE_ENUM_FLAGS(MaterialFeatureFlags)

struct MaterialPipelineDesc
{
    MaterialProgramFamily programFamily = MaterialProgramFamily::StandardLit;
    MaterialFeatureFlags featureFlags   = MaterialFeatureFlags::None;
    float alphaCutoff                   = 0.5f;
};

struct MeshResourceDesc
{
    struct Vertex
    {
        Diligent::float3 position{};
        Diligent::float3 normal{0.0f, 1.0f, 0.0f};
        float texCoordU = 0.0f;
        float texCoordV = 0.0f;
        Diligent::float4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    };

    std::string debugName;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct MaterialResourceDesc
{
    std::string debugName;
    Diligent::float3 baseColor{1.0f, 1.0f, 1.0f};
    float metallic  = 0.0f;
    float roughness = 0.5f;
    Diligent::float3 emissiveFactor{0.0f, 0.0f, 0.0f};
    TextureHandle baseColorTexture{};
    TextureHandle normalTexture{};
    TextureHandle metallicRoughnessTexture{};
    TextureHandle emissiveTexture{};
    TextureHandle aoTexture{};
    MaterialPipelineDesc pipeline{};
    MaterialRenderMode renderMode = MaterialRenderMode::Opaque;
    // Orders materials only within the same render mode; lower values draw earlier.
    std::int32_t renderOrder      = 0;
    float opacity                 = 1.0f;
    bool castsShadows             = true;
    bool receivesShadows          = true;
};

inline bool usesTransparentPass(const MaterialResourceDesc &desc) noexcept
{
    return desc.renderMode == MaterialRenderMode::Transparent;
}

inline MaterialFeatureFlags effectiveMaterialFeatureFlags(const MaterialResourceDesc &desc) noexcept
{
    MaterialFeatureFlags flags = desc.pipeline.featureFlags;
    constexpr std::uint32_t kAlphaTestBit =
        static_cast<std::uint32_t>(MaterialFeatureFlags::AlphaTest);
    std::uint32_t rawFlags = static_cast<std::uint32_t>(flags);
    if (desc.renderMode == MaterialRenderMode::Cutout)
    {
        rawFlags |= kAlphaTestBit;
    }
    else
    {
        rawFlags &= ~kAlphaTestBit;
    }
    return static_cast<MaterialFeatureFlags>(rawFlags);
}

struct TextureResourceDesc
{
    struct SubresourceDesc
    {
        std::vector<std::uint8_t> pixelData;
    };

    std::string debugName;
    std::uint32_t width            = 1u;
    std::uint32_t height           = 1u;
    std::uint32_t mipLevelCount    = 1u;
    TextureDimension dimension     = TextureDimension::Texture2D;
    TexturePixelFormat pixelFormat = TexturePixelFormat::RGBA8;
    TextureColorSpace colorSpace   = TextureColorSpace::Linear;
    TextureMipPolicy mipPolicy     = TextureMipPolicy::Disabled;
    std::vector<SubresourceDesc> subresources;
    std::vector<std::uint8_t> pixelData;
};

struct EnvironmentIblDesc
{
    TextureHandle backgroundCubemap{};
    TextureHandle irradianceCubemap{};
    TextureHandle prefilteredSpecularCubemap{};
    float intensity           = 1.0f;
    float backgroundIntensity = 1.0f;

    bool enabled(IblQualityTier tier) const noexcept
    {
        switch (tier)
        {
        case IblQualityTier::Off:
            return false;
        case IblQualityTier::DiffuseOnly:
            return irradianceCubemap.id != common::kInvalidResourceId;
        case IblQualityTier::Full:
            return irradianceCubemap.id != common::kInvalidResourceId &&
                   prefilteredSpecularCubemap.id != common::kInvalidResourceId;
        default:
            return false;
        }
    }
};

struct EnvironmentFluidDesc
{
    float smoothness                = 0.92f;
    Diligent::float3 specular       = {0.35f, 0.4f, 0.45f};
    float fresnel                   = 0.8f;
    float depthEdgeThreshold        = 0.2f;
    float filterRadiusPixels        = 6.0f;
    float filterWorldRadius         = 0.18f;
    float filterDepthThreshold      = 0.12f;
    bool enableBackgroundRefraction = true;
    float refractionIor             = 1.33f;
    float refractionViewThickness   = 0.35f;
};

inline EnvironmentFluidDesc defaultEnvironmentFluidDesc() noexcept
{
    return EnvironmentFluidDesc{};
}

class CRESSIM_NEO_GRAPHICS_API RenderResourceManager
{
public:
    RenderResourceManager();
    ~RenderResourceManager();

    RenderResourceManager(const RenderResourceManager &other);
    RenderResourceManager &operator=(const RenderResourceManager &other);
    RenderResourceManager(RenderResourceManager &&other) noexcept;
    RenderResourceManager &operator=(RenderResourceManager &&other) noexcept;

    MeshHandle registerMesh(const MeshResourceDesc &desc);
    MaterialHandle registerMaterial(const MaterialResourceDesc &desc);
    TextureHandle registerTexture(const TextureResourceDesc &desc);

    bool isValid(MeshHandle mesh) const;
    bool isValid(MaterialHandle material) const;
    bool isValid(TextureHandle texture) const;

    const MeshResourceDesc *tryGetMesh(MeshHandle mesh) const noexcept;
    const MaterialResourceDesc *tryGetMaterial(MaterialHandle material) const noexcept;
    const TextureResourceDesc *tryGetTexture(TextureHandle texture) const noexcept;
    bool tryGetMeshLocalBounds(MeshHandle mesh, Diligent::float3 &outMin,
                               Diligent::float3 &outMax) const noexcept;

    std::uint64_t meshVersion(MeshHandle mesh) const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
