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

/// @file render_resource_manager.h
/// @brief Host mesh, material, and texture resource definitions and asset registration manager.

namespace cressim::neo::graphics
{

/// @brief Handle referencing a registered CPU/GPU triangle mesh asset.
struct MeshHandle
{
    common::ResourceId id = common::kInvalidResourceId; ///< Unique numeric resource identifier.
};

/// @brief Handle referencing a registered surface material asset.
struct MaterialHandle
{
    common::ResourceId id = common::kInvalidResourceId; ///< Unique numeric resource identifier.
};

/// @brief Handle referencing a registered 2D or cubemap texture asset.
struct TextureHandle
{
    common::ResourceId id = common::kInvalidResourceId; ///< Unique numeric resource identifier.
};

/// @brief Shader program specialization family applied when rendering surface primitives.
enum class MaterialProgramFamily
{
    StandardLit, ///< Standard PBR metallic-roughness lit surface pipeline.
    SoftBodyLit, ///< Deformable tetrahedral/surface soft-body mesh pipeline with skinning.
    CurveLit,    ///< Elastic strand and ribbon curve rendering pipeline.
};

/// @brief Blending and rasterization mode determining pipeline stage assignment.
enum class MaterialRenderMode
{
    Opaque,      ///< Fully opaque surface writing depth and color in the primary pass.
    Cutout,      ///< Alpha-tested masked surface discarding pixels below cutoff threshold.
    Transparent, ///< Alpha-blended transparent surface rendered in the back-to-front pass.
};

/// @brief Color space encoding used when sampling texture data.
enum class TextureColorSpace
{
    Linear, ///< Linear color or non-color data (normal maps, metallic-roughness, ambient
            ///< occlusion).
    Srgb,   ///< sRGB non-linear gamma-corrected color data (base color, albedo, emissive).
};

/// @brief Channel layout and bit depth of pixel buffers.
enum class TexturePixelFormat
{
    RGBA8,   ///< 8-bit unsigned normalized RGBA (4 bytes per pixel).
    RGBA16F, ///< 16-bit floating-point half-precision RGBA (8 bytes per pixel, HDR).
};

/// @brief Mipmap generation policy upon texture upload.
enum class TextureMipPolicy
{
    Disabled, ///< Only base mip level (0) is loaded; no mipmaps are generated.
    Generate, ///< Full mipmap chain is generated on the GPU upon upload.
};

/// @brief Spatial dimension category of a texture asset.
enum class TextureDimension
{
    Texture2D,   ///< Standard 2D planar texture.
    TextureCube, ///< 6-face cubemap texture for skyboxes and image-based lighting.
};

/// @brief Image-based lighting (IBL) environment quality tier.
enum class IblQualityTier : std::uint32_t
{
    Off         = 0u, ///< Environment IBL lighting disabled.
    DiffuseOnly = 1u, ///< Diffuse irradiance spherical harmonics / irradiance cubemap only.
    Full        = 2u, ///< Full diffuse irradiance and split-sum prefiltered specular reflections.
};

/// @brief Bitmask flags enabling material shader features and extended microfacet lobes.
enum class MaterialFeatureFlags : std::uint32_t
{
    None        = 0u,       ///< No optional shader features enabled.
    AlphaTest   = 1u << 0u, ///< Alpha testing / cutout mask discarding enabled.
    NormalMap   = 1u << 1u, ///< Tangent-space normal mapping enabled.
    ClearCoat   = 1u << 2u, ///< Secondary clear coat specular layer enabled.
    DoubleSided = 1u << 3u, ///< Disables backface culling and flips normals on backfaces.
};
CRESSIM_NEO_DEFINE_ENUM_FLAGS(MaterialFeatureFlags)

/// @brief Pipeline shader configuration for compiling or selecting material PSO variants.
struct MaterialPipelineDesc
{
    MaterialProgramFamily programFamily = MaterialProgramFamily::StandardLit; ///< Shader family.
    MaterialFeatureFlags featureFlags   = MaterialFeatureFlags::None;         ///< Feature bitmask.
    float alphaCutoff                   = 0.5f; ///< Alpha discard threshold for Cutout mode.
};

/// @brief Geometric vertex and index data definition for registering a mesh resource.
struct MeshResourceDesc
{
    /// @brief Interleaved vertex attribute layout.
    struct Vertex
    {
        Diligent::float3 position{};               ///< 3D object-space position.
        Diligent::float3 normal{0.0f, 1.0f, 0.0f}; ///< Surface normal vector.
        float texCoordU = 0.0f;                    ///< Horizontal UV texture coordinate.
        float texCoordV = 0.0f;                    ///< Vertical UV texture coordinate.
        Diligent::float4 tangent{1.0f, 0.0f, 0.0f,
                                 1.0f}; ///< Tangent vector (xyz) and bitangent sign (w).
    };

    std::string debugName;              ///< Human-readable debug name.
    std::vector<Vertex> vertices;       ///< Interleaved vertex attribute list.
    std::vector<std::uint32_t> indices; ///< Triangle index buffer (3 indices per face).
};

/// @brief Physical surface material description including PBR textures and parameters.
struct MaterialResourceDesc
{
    std::string debugName;                             ///< Human-readable debug identifier.
    Diligent::float3 baseColor{1.0f, 1.0f, 1.0f};      ///< Constant base color multiplier (RGB).
    float metallic  = 0.0f;                            ///< Metallic reflection factor [0..1].
    float roughness = 0.5f;                            ///< Microfacet roughness factor [0..1].
    Diligent::float3 emissiveFactor{0.0f, 0.0f, 0.0f}; ///< Emissive color radiance (RGB).
    TextureHandle baseColorTexture{};                  ///< Base color / albedo texture map.
    TextureHandle normalTexture{};                     ///< Tangent-space normal map.
    TextureHandle
        metallicRoughnessTexture{};  ///< Combined metallic (B) and roughness (G) texture map.
    TextureHandle emissiveTexture{}; ///< Emissive radiance texture map.
    TextureHandle aoTexture{};       ///< Precomputed ambient occlusion (R) map.
    MaterialPipelineDesc pipeline{}; ///< Pipeline program family and feature flags.
    MaterialRenderMode renderMode =
        MaterialRenderMode::Opaque; ///< Rasterization pass blending mode.
    std::int32_t renderOrder =
        0; ///< Sorting order within the same render mode bucket (lower draws earlier).
    float opacity        = 1.0f; ///< Surface opacity factor [0..1].
    bool castsShadows    = true; ///< Whether objects with this material render into shadow maps.
    bool receivesShadows = true; ///< Whether surface samples shadow maps during shading.
};

/// @brief Checks if a material is configured to draw in the transparent render pass.
/// @param desc Material descriptor to check.
/// @return True if renderMode is Transparent.
inline bool usesTransparentPass(const MaterialResourceDesc &desc) noexcept
{
    return desc.renderMode == MaterialRenderMode::Transparent;
}

/// @brief Derives the effective shader feature flags for a material based on its mode and pipeline
/// settings.
/// @param desc Material descriptor.
/// @return Adjusted MaterialFeatureFlags bitmask.
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

/// @brief Texture asset descriptor containing image dimensions, formats, and pixel data payloads.
struct TextureResourceDesc
{
    /// @brief Subresource byte payload for an individual mip level or cubemap face.
    struct SubresourceDesc
    {
        std::vector<std::uint8_t> pixelData; ///< Subresource pixel bytes.
    };

    std::string debugName;               ///< Debug identifier.
    std::uint32_t width            = 1u; ///< Image width in pixels.
    std::uint32_t height           = 1u; ///< Image height in pixels.
    std::uint32_t mipLevelCount    = 1u; ///< Number of mipmap levels present.
    TextureDimension dimension     = TextureDimension::Texture2D; ///< 2D or Cubemap dimensionality.
    TexturePixelFormat pixelFormat = TexturePixelFormat::RGBA8;   ///< Pixel data format.
    TextureColorSpace colorSpace   = TextureColorSpace::Linear;   ///< Color space interpretation.
    TextureMipPolicy mipPolicy =
        TextureMipPolicy::Disabled;            ///< Automatic mipmap generation behavior.
    std::vector<SubresourceDesc> subresources; ///< Array of explicit subresource payloads.
    std::vector<std::uint8_t> pixelData;       ///< Contiguous raw pixel payload for level 0.
};

/// @brief Image-based lighting (IBL) environment maps and radiance intensity multipliers.
struct EnvironmentIblDesc
{
    TextureHandle backgroundCubemap{}; ///< Panoramic or cubemap background environment skybox.
    TextureHandle irradianceCubemap{}; ///< Diffuse Lambertian irradiance convolution cubemap.
    TextureHandle
        prefilteredSpecularCubemap{}; ///< Split-sum prefiltered specular reflection cubemap.
    float intensity           = 1.0f; ///< Ambient radiance intensity multiplier.
    float backgroundIntensity = 1.0f; ///< Background skybox display intensity multiplier.

    /// @brief Checks if IBL resources are available and active for the requested quality tier.
    /// @param tier Target IBL quality tier.
    /// @return True if required cubemaps are valid.
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

/// @brief Screen-space fluid surface rendering and optical refraction parameters.
struct EnvironmentFluidDesc
{
    float smoothness           = 0.92f;                ///< Surface smoothness parameter.
    Diligent::float3 specular  = {0.35f, 0.4f, 0.45f}; ///< Specular highlight tint color.
    float fresnel              = 0.8f;                 ///< Fresnel reflection exponent.
    float depthEdgeThreshold   = 0.2f;                 ///< Depth discontinuity edge threshold.
    float filterRadiusPixels   = 6.0f;  ///< Bilateral depth smoothing filter radius in pixels.
    float filterWorldRadius    = 0.18f; ///< World-space depth filter cutoff distance.
    float filterDepthThreshold = 0.12f; ///< Depth gradient bilateral threshold.
    bool enableBackgroundRefraction =
        true; ///< Enable background scene refraction distorting through the fluid volume.
    float refractionIor           = 1.33f; ///< Index of refraction (1.33 for water).
    float refractionViewThickness = 0.35f; ///< Assumed optical thickness along camera view vector.
};

/// @brief Returns default rendering settings for fluid surfaces.
/// @return EnvironmentFluidDesc struct with default water-like parameters.
inline EnvironmentFluidDesc defaultEnvironmentFluidDesc() noexcept
{
    return EnvironmentFluidDesc{};
}

/// @brief Central repository managing host-side registration and lookup for meshes, materials, and
/// textures.
class CRESSIM_NEO_GRAPHICS_API RenderResourceManager
{
public:
    /// @brief Default constructor.
    RenderResourceManager();

    /// @brief Destructor.
    ~RenderResourceManager();

    RenderResourceManager(const RenderResourceManager &other);
    RenderResourceManager &operator=(const RenderResourceManager &other);
    RenderResourceManager(RenderResourceManager &&other) noexcept;
    RenderResourceManager &operator=(RenderResourceManager &&other) noexcept;

    /// @brief Registers a new mesh asset and allocates a unique handle.
    ///
    /// Missing or invalid tangent data is generated or repaired in the stored descriptor.
    /// @param desc Mesh geometry descriptor.
    /// @return Allocated MeshHandle.
    MeshHandle registerMesh(const MeshResourceDesc &desc);

    /// @brief Registers a new surface material and allocates a unique handle.
    ///
    /// The stored descriptor derives AlphaTest from Cutout mode and enables NormalMap when a
    /// normal texture is supplied.
    /// @param desc Material descriptor.
    /// @return Allocated MaterialHandle.
    MaterialHandle registerMaterial(const MaterialResourceDesc &desc);

    /// @brief Registers a new texture and allocates a unique handle.
    ///
    /// Stored width, height, and mip count are clamped to at least one. A non-empty pixelData
    /// payload is copied into the first subresource when explicit subresources are absent.
    /// @param desc Texture descriptor.
    /// @return Allocated TextureHandle.
    TextureHandle registerTexture(const TextureResourceDesc &desc);

    /// @brief Validates whether a mesh handle refers to an existing mesh resource.
    /// @param mesh Handle to check.
    /// @return True if valid.
    bool isValid(MeshHandle mesh) const;

    /// @brief Validates whether a material handle refers to an existing material resource.
    /// @param material Handle to check.
    /// @return True if valid.
    bool isValid(MaterialHandle material) const;

    /// @brief Validates whether a texture handle refers to an existing texture resource.
    /// @param texture Handle to check.
    /// @return True if valid.
    bool isValid(TextureHandle texture) const;

    /// @brief Looks up the mesh resource descriptor for a valid handle.
    /// @param mesh Handle to query.
    /// @return Pointer to MeshResourceDesc or nullptr if invalid.
    const MeshResourceDesc *tryGetMesh(MeshHandle mesh) const noexcept;

    /// @brief Looks up the material resource descriptor for a valid handle.
    /// @param material Handle to query.
    /// @return Pointer to MaterialResourceDesc or nullptr if invalid.
    const MaterialResourceDesc *tryGetMaterial(MaterialHandle material) const noexcept;

    /// @brief Looks up the texture resource descriptor for a valid handle.
    /// @param texture Handle to query.
    /// @return Pointer to TextureResourceDesc or nullptr if invalid.
    const TextureResourceDesc *tryGetTexture(TextureHandle texture) const noexcept;

    /// @brief Computes the local axis-aligned bounding box for a registered mesh.
    /// @param mesh Mesh handle.
    /// @param outMin Output minimum corner (xyz).
    /// @param outMax Output maximum corner (xyz).
    /// @return True if mesh was found and bounds computed.
    bool tryGetMeshLocalBounds(MeshHandle mesh, Diligent::float3 &outMin,
                               Diligent::float3 &outMax) const noexcept;

    /// @brief Retrieves the registration version for a mesh asset.
    /// @param mesh Mesh handle.
    /// @return One for a registered mesh, or zero for an invalid handle.
    std::uint64_t meshVersion(MeshHandle mesh) const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDER_RESOURCE_MANAGER_H
