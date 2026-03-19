#ifndef CRESSIM_NEO_GPU_GPU_SCENE_H
#define CRESSIM_NEO_GPU_GPU_SCENE_H

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <array>
#include <cstdint>

namespace cressim::neo::gpu
{

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

struct GpuPoseBufferView
{
    Diligent::IBuffer* positionsBuffer    = nullptr;
    Diligent::IBuffer* orientationsBuffer = nullptr;
    Diligent::IBuffer* scalesBuffer       = nullptr;
    std::uint32_t count                   = 0;
};

struct GpuCameraInput
{
    Diligent::float4 position{};
    Diligent::float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float4 projectionParams{};
    std::uint32_t envIndex   = 0u;
    std::uint32_t cameraSlot = 0u;
    std::uint32_t active     = 0u;
    std::uint32_t reserved   = 0u;
};

struct GpuDirectionalLightInput
{
    Diligent::float4 directionIntensity{};
    Diligent::float4 color{};
    Diligent::float4 shadowParams{};
    std::uint32_t envIndex  = 0u;
    std::uint32_t lightSlot = 0u;
    std::uint32_t active    = 0u;
    std::uint32_t reserved  = 0u;
};

struct GpuPreparedCamera
{
    Diligent::float4x4 viewMatrix           = Diligent::float4x4::Identity();
    Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
    std::array<Diligent::float4x4, 4> lightViewProjectionMatrices{};
    Diligent::float4 cameraPosition{};
    Diligent::float4 cascadeSplits{};
    Diligent::float4 shadowParams{};
    std::uint32_t envIndex             = 0u;
    std::uint32_t active               = 0u;
    std::uint32_t objectRangeStart     = 0u;
    std::uint32_t objectRangeCount     = 0u;
    std::uint32_t visibilityDataOffset = 0u;
    std::uint32_t reserved0            = 0u;
    std::uint32_t reserved1            = 0u;
    std::uint32_t reserved2            = 0u;
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
    Diligent::IBuffer* renderableMetadataBuffer           = nullptr;
    Diligent::IBuffer* renderableQueueInfoBuffer          = nullptr;
    Diligent::IBuffer* renderableVisibilityFlagsBuffer    = nullptr;
    Diligent::IBuffer* renderableShadowCascadeMasksBuffer = nullptr;
    Diligent::IBuffer* cameraInputsBuffer                 = nullptr;
    Diligent::IBuffer* preparedCamerasBuffer              = nullptr;
    Diligent::IBuffer* lightInputsBuffer                  = nullptr;
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
