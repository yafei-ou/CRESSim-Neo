#ifndef CRESSIM_NEO_GRAPHICS_GPU_SCENE_PIPELINE_TYPES_H
#define CRESSIM_NEO_GRAPHICS_GPU_SCENE_PIPELINE_TYPES_H

#include "common/scene_primitives.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <array>
#include <cstdint>

/// @file gpu_scene_pipeline_types.h
/// @brief GPU scene layout constants, shadow view parameters, prepared camera uniforms, and
/// visibility pair structures.

namespace cressim::neo::graphics
{

/// @brief Sentinel constant representing an invalid or unallocated GPU scene slot index.
inline constexpr std::uint32_t kInvalidGpuSceneIndex     = 0xffffffffu;
/// @brief Fixed slot index within each environment reserved for the main directional sun light.
inline constexpr std::uint32_t kMainDirectionalLightSlot = 0u;
/// @brief Sentinel constant representing an unassigned batch camera shadow layer.
inline constexpr std::uint32_t kInvalidBatchCameraLayer  = 0xffffffffu;

/// @brief Maximum number of local (point/spot) lights processed per environment during forward
/// shading.
inline constexpr std::uint32_t kForwardLocalLightCap    = 8u;
/// @brief Maximum number of shadowed local lights evaluated per environment.
inline constexpr std::uint32_t kShadowedLocalLightCap   = 4u;
/// @brief Maximum number of active shadowed omnidirectional point lights.
inline constexpr std::uint32_t kShadowedPointLightCap   = 1u;
/// @brief Maximum cubemap face count for point light omnidirectional shadow maps.
inline constexpr std::uint32_t kLocalShadowMaxFaceCount = 6u;

/// @brief Computes the global linear light index for the primary directional light in a given
/// environment.
/// @param layout Multi-environment scene layout descriptor.
/// @param envIndex Zero-based environment index.
/// @return Global GPU light buffer index or kInvalidGpuSceneIndex.
constexpr std::uint32_t mainDirectionalLightIndex(const common::SceneLayoutDesc &layout,
                                                  std::uint32_t envIndex) noexcept
{
    return layout.maxLightsPerEnv == 0u
               ? kInvalidGpuSceneIndex
               : envIndex * layout.maxLightsPerEnv + kMainDirectionalLightSlot;
}

/// @brief Shadow casting and projection mode for a GPU light source.
enum class GpuLightShadowMode : std::uint32_t
{
    None    = 0u, ///< Light does not cast shadows.
    Local2D = 1u, ///< 2D spotlight projection shadow map.
    Point   = 2u, ///< 6-face omnidirectional cubemap point shadow map.
};

/// @brief Mapping linking a scene light to its allocated shadow map viewport and projection slice.
struct GpuLightShadowAssignment
{
    std::uint32_t shadowMode =
        static_cast<std::uint32_t>(GpuLightShadowMode::None); ///< Active shadow mode enum.
    std::uint32_t shadowViewIndex =
        kInvalidGpuSceneIndex;    ///< Index into global local shadow view array.
    std::uint32_t reserved0 = 0u; ///< Reserved padding.
    std::uint32_t reserved1 = 0u; ///< Reserved padding.
};

/// @brief GPU uniform buffer payload describing a local (spot/point) shadow projection camera.
struct GpuLocalShadowView
{
    std::array<Diligent::float4x4, kLocalShadowMaxFaceCount>
        lightViewProjectionMatrices{}; ///< View-projection matrices for spot or 6 cube faces.
    Diligent::float4
        lightPositionRange{}; ///< World-space light position (xyz) and attenuation range (w).
    Diligent::float4 lightDirection{}; ///< World-space light direction vector.
    Diligent::float2 shadowTexelSize{
        0.0f, 0.0f};              ///< Reciprocal shadow map dimensions (1/width, 1/height).
    float shadowNearPlane = 0.0f; ///< Near clipping plane distance.
    float shadowFarPlane  = 0.0f; ///< Far clipping plane distance.
    std::uint32_t lightIndex =
        kInvalidGpuSceneIndex;     ///< Associated light index in the scene light buffer.
    std::uint32_t envIndex   = 0u; ///< Associated environment index.
    std::uint32_t firstLayer = 0u; ///< First shadow map texture array slice.
    std::uint32_t layerCount = 1u; ///< Number of shadow texture array slices.
    std::uint32_t lightType  = 0u; ///< Light type identifier.
    std::uint32_t active     = 0u; ///< Active flag (1 = active, 0 = inactive).
    std::uint32_t reserved0  = 0u; ///< Reserved padding.
    std::uint32_t reserved1  = 0u; ///< Reserved padding.
};

/// @brief GPU uniform buffer structure holding transformed camera matrices, cascaded shadow maps,
/// and culling ranges.
struct GpuPreparedCamera
{
    Diligent::float4x4 viewMatrix =
        Diligent::float4x4::Identity(); ///< World-to-camera view transformation matrix.
    Diligent::float4x4 viewProjectionMatrix =
        Diligent::float4x4::Identity(); ///< Combined view-projection matrix.
    std::array<Diligent::float4x4, 4>
        lightViewProjectionMatrices{}; ///< View-projection matrices for up to 4 directional shadow
                                       ///< cascades.
    Diligent::float4 cameraPosition{}; ///< World-space camera eye position.
    Diligent::float4 cascadeSplits{};  ///< Depth split distances for cascaded shadow mapping.
    Diligent::float2 mainShadowTexelSize{0.0f, 0.0f}; ///< Directional shadow map texel size.
    float mainShadowCascadeCount       = 0.0f; ///< Number of active directional shadow cascades.
    float mainShadowFadeDistance       = 0.0f; ///< Shadow fade-out distance.
    std::uint32_t envIndex             = 0u;   ///< Environment index this camera renders.
    std::uint32_t active               = 0u;   ///< Active flag (1 = active, 0 = inactive).
    std::uint32_t objectRangeStart     = 0u;   ///< Starting index into object draw list.
    std::uint32_t objectRangeCount     = 0u;   ///< Number of candidate objects for this camera.
    std::uint32_t visibilityDataOffset = 0u;   ///< Offset into visibility bitmask buffer.
    std::uint32_t reserved0            = 0u;   ///< Reserved padding.
    std::uint32_t reserved1            = 0u;   ///< Reserved padding.
    std::uint32_t reserved2            = 0u;   ///< Reserved padding.
};

/// @brief Routing metadata describing an active camera in a multi-camera batched render pass.
struct GpuBatchCameraMetadata
{
    std::uint32_t globalCameraIndex = 0u; ///< Global camera index across all environments.
    std::uint32_t envIndex          = 0u; ///< Parent environment index.
    std::uint32_t mainLightIndex =
        kInvalidGpuSceneIndex; ///< Primary directional light index for this camera's environment.
    std::uint32_t colorLayer = 0u; ///< Destination color texture array layer.
    std::uint32_t shadowLayer =
        kInvalidBatchCameraLayer; ///< Destination shadow texture array layer.
    std::uint32_t reserved0 = 0u; ///< Reserved padding.
    std::uint32_t reserved1 = 0u; ///< Reserved padding.
    std::uint32_t reserved2 = 0u; ///< Reserved padding.
};

/// @brief Output record produced by GPU frustum culling representing a visible (object, camera)
/// pair.
struct GpuVisiblePairInstance
{
    std::uint32_t objectIndex = 0u; ///< Index of visible render object in the global object table.
    std::uint32_t batchCameraIndex   = 0u; ///< Index of the camera observing the object.
    std::uint32_t bucketIndex        = 0u; ///< Material/pipeline sort bucket index.
    std::uint32_t shadowSubviewIndex = 0u; ///< Shadow cascade or face subview index.
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GPU_SCENE_PIPELINE_TYPES_H
