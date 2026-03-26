#ifndef CRESSIM_NEO_GPU_GPU_SCENE_H
#define CRESSIM_NEO_GPU_GPU_SCENE_H

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <array>
#include <cstdint>

namespace cressim::neo::gpu
{

inline constexpr std::uint32_t kInvalidGpuSceneIndex     = 0xffffffffu;
inline constexpr std::uint32_t kMainDirectionalLightSlot = 0u;
inline constexpr std::uint32_t kInvalidBatchCameraLayer  = 0xffffffffu;
inline constexpr std::uint32_t kForwardLocalLightCap     = 8u;
inline constexpr std::uint32_t kShadowedLocalLightCap    = 4u;
inline constexpr std::uint32_t kShadowedPointLightCap    = 1u;
inline constexpr std::uint32_t kLocalShadowMaxFaceCount  = 6u;

struct GpuSceneLayoutDesc
{
    std::uint32_t envCount         = 1u;
    std::uint32_t maxObjectsPerEnv = 4096u;
    std::uint32_t maxLightsPerEnv  = 4u;
    std::uint32_t maxCamerasPerEnv = 4u;

    [[nodiscard]] std::uint32_t totalObjectCapacity() const noexcept
    {
        return envCount * maxObjectsPerEnv;
    }
    [[nodiscard]] std::uint32_t totalLightCapacity() const noexcept
    {
        return envCount * maxLightsPerEnv;
    }
    [[nodiscard]] std::uint32_t totalCameraCapacity() const noexcept
    {
        return envCount * maxCamerasPerEnv;
    }
};

[[nodiscard]] constexpr std::uint32_t mainDirectionalLightIndex(const GpuSceneLayoutDesc &layout,
                                                                std::uint32_t envIndex) noexcept
{
    return layout.maxLightsPerEnv == 0u
               ? kInvalidGpuSceneIndex
               : envIndex * layout.maxLightsPerEnv + kMainDirectionalLightSlot;
}

struct GpuPoseBufferView
{
    Diligent::IBuffer *positionsBuffer    = nullptr;
    Diligent::IBuffer *orientationsBuffer = nullptr;
    Diligent::IBuffer *scalesBuffer       = nullptr;
    std::uint32_t count                   = 0;
};

struct GpuCameraInput
{
    Diligent::float4 position{};
    Diligent::float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float4 projectionParams{60.0f, 0.01f, 1000.0f, 0.0f};
    Diligent::float4 viewportAndOutputSize{0.0f, 0.0f, 1.0f, 1.0f};
    std::uint32_t envIndex   = 0u;
    std::uint32_t cameraSlot = 0u;
    std::uint32_t active     = 0u;
    std::uint32_t reserved   = 0u;
};

struct GpuLightInput
{
    Diligent::float4 positionRange{};
    Diligent::float4 directionIntensity{};
    Diligent::float4 color{};
    Diligent::float4 spotAngles{};
    float shadowDistance       = 0.0f;
    float shadowFadeDistance   = 0.0f;
    float shadowBias           = 0.0015f;
    float shadowPadding0       = 0.0f;
    std::uint32_t envIndex     = 0u;
    std::uint32_t lightSlot    = 0u;
    std::uint32_t type         = 0u;
    std::uint32_t active       = 0u;
    std::uint32_t castsShadows = 0u;
    std::uint32_t reserved0    = 0u;
    std::uint32_t reserved1    = 0u;
    std::uint32_t reserved2    = 0u;
};

enum class GpuLightType : std::uint32_t
{
    Directional = 0u,
    Point       = 1u,
    Spot        = 2u,
};

struct GpuLocalLightSelection
{
    std::uint32_t localLightCount         = 0u;
    std::uint32_t shadowedLocalLightCount = 0u;
    std::uint32_t shadowedPointLightCount = 0u;
    std::uint32_t reserved0               = 0u;
    std::array<std::uint32_t, kForwardLocalLightCap> lightIndices{};
};

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
    std::uint32_t objectIndex      = 0u;
    std::uint32_t batchCameraIndex = 0u;
    std::uint32_t bucketIndex      = 0u;
    std::uint32_t reserved0        = 0u;
};

struct GpuRenderableQueueInfo
{
    std::uint32_t opaqueCommandIndex     = 0xffffffffu;
    std::uint32_t shadowCommandBaseIndex = 0xffffffffu;
    std::uint32_t reserved0              = 0u;
    std::uint32_t reserved1              = 0u;
};

struct GpuEntitySceneView
{
    GpuSceneLayoutDesc layout{};
    GpuPoseBufferView poses{};
    Diligent::IBuffer *renderableMetadataBuffer           = nullptr;
    Diligent::IBuffer *renderableQueueInfoBuffer          = nullptr;
    Diligent::IBuffer *renderableVisibilityFlagsBuffer    = nullptr;
    Diligent::IBuffer *renderableShadowCascadeMasksBuffer = nullptr;
    Diligent::IBuffer *cameraInputsBuffer                 = nullptr;
    Diligent::IBuffer *preparedCamerasBuffer              = nullptr;
    Diligent::IBuffer *lightInputsBuffer                  = nullptr;
    Diligent::IBuffer *localLightSelectionBuffer          = nullptr;
    std::uint32_t entityCount                             = 0;
    std::uint32_t renderableCount                         = 0;
    std::uint32_t cameraCount                             = 0;
    std::uint32_t lightCount                              = 0;
};

struct GpuEntityPoseMappingEntry
{
    std::uint32_t sourcePoseIndex = 0;
    std::uint32_t objectIndex     = 0;
};

enum GpuRenderableFlags : std::uint32_t
{
    GpuRenderableFlag_None         = 0u,
    GpuRenderableFlag_Active       = 1u << 0u,
    GpuRenderableFlag_Opaque       = 1u << 1u,
    GpuRenderableFlag_ShadowCaster = 1u << 2u,
};

struct GpuRenderableMetadata
{
    std::uint32_t flags     = GpuRenderableFlag_None;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::uint32_t reserved2 = 0u;
    Diligent::float4 localBoundsMin{};
    Diligent::float4 localBoundsMax{};
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_SCENE_H
