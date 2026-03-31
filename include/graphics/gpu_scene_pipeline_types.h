#ifndef CRESSIM_NEO_GRAPHICS_GPU_SCENE_PIPELINE_TYPES_H
#define CRESSIM_NEO_GRAPHICS_GPU_SCENE_PIPELINE_TYPES_H

#include "common/scene_primitives.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <array>
#include <cstdint>

namespace cressim::neo::graphics
{

inline constexpr std::uint32_t kInvalidGpuSceneIndex     = 0xffffffffu;
inline constexpr std::uint32_t kMainDirectionalLightSlot = 0u;
inline constexpr std::uint32_t kInvalidBatchCameraLayer  = 0xffffffffu;
inline constexpr std::uint32_t kForwardLocalLightCap     = 8u;
inline constexpr std::uint32_t kShadowedLocalLightCap    = 4u;
inline constexpr std::uint32_t kShadowedPointLightCap    = 1u;
inline constexpr std::uint32_t kLocalShadowMaxFaceCount  = 6u;

constexpr std::uint32_t mainDirectionalLightIndex(const common::SceneLayoutDesc &layout,
                                                  std::uint32_t envIndex) noexcept
{
    return layout.maxLightsPerEnv == 0u
               ? kInvalidGpuSceneIndex
               : envIndex * layout.maxLightsPerEnv + kMainDirectionalLightSlot;
}

enum class GpuLightShadowMode : std::uint32_t
{
    None    = 0u,
    Local2D = 1u,
    Point   = 2u,
};

struct GpuLightShadowAssignment
{
    std::uint32_t shadowMode      = static_cast<std::uint32_t>(GpuLightShadowMode::None);
    std::uint32_t shadowViewIndex = kInvalidGpuSceneIndex;
    std::uint32_t reserved0       = 0u;
    std::uint32_t reserved1       = 0u;
};

struct GpuLocalShadowView
{
    std::array<Diligent::float4x4, kLocalShadowMaxFaceCount> lightViewProjectionMatrices{};
    Diligent::float4 lightPositionRange{};
    Diligent::float4 lightDirection{};
    Diligent::float2 shadowTexelSize{0.0f, 0.0f};
    float shadowNearPlane    = 0.0f;
    float shadowFarPlane     = 0.0f;
    std::uint32_t lightIndex = kInvalidGpuSceneIndex;
    std::uint32_t envIndex   = 0u;
    std::uint32_t firstLayer = 0u;
    std::uint32_t layerCount = 1u;
    std::uint32_t lightType  = 0u;
    std::uint32_t active     = 0u;
    std::uint32_t reserved0  = 0u;
    std::uint32_t reserved1  = 0u;
};

struct GpuPreparedCamera
{
    Diligent::float4x4 viewMatrix           = Diligent::float4x4::Identity();
    Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
    std::array<Diligent::float4x4, 4> lightViewProjectionMatrices{};
    Diligent::float4 cameraPosition{};
    Diligent::float4 cascadeSplits{};
    Diligent::float2 mainShadowTexelSize{0.0f, 0.0f};
    float mainShadowCascadeCount       = 0.0f;
    float mainShadowFadeDistance       = 0.0f;
    std::uint32_t envIndex             = 0u;
    std::uint32_t active               = 0u;
    std::uint32_t objectRangeStart     = 0u;
    std::uint32_t objectRangeCount     = 0u;
    std::uint32_t visibilityDataOffset = 0u;
    std::uint32_t reserved0            = 0u;
    std::uint32_t reserved1            = 0u;
    std::uint32_t reserved2            = 0u;
};

struct GpuBatchCameraMetadata
{
    std::uint32_t globalCameraIndex = 0u;
    std::uint32_t envIndex          = 0u;
    std::uint32_t mainLightIndex    = kInvalidGpuSceneIndex;
    std::uint32_t colorLayer        = 0u;
    std::uint32_t shadowLayer       = kInvalidBatchCameraLayer;
    std::uint32_t reserved0         = 0u;
    std::uint32_t reserved1         = 0u;
    std::uint32_t reserved2         = 0u;
};

struct GpuVisiblePairInstance
{
    std::uint32_t objectIndex        = 0u;
    std::uint32_t batchCameraIndex   = 0u;
    std::uint32_t bucketIndex        = 0u;
    std::uint32_t shadowSubviewIndex = 0u;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GPU_SCENE_PIPELINE_TYPES_H
