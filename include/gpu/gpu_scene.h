#ifndef CRESSIM_NEO_GPU_GPU_SCENE_H
#define CRESSIM_NEO_GPU_GPU_SCENE_H

#include "gpu/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <cstdint>

namespace cressim::neo::gpu
{

struct GpuPoseBufferView
{
    Diligent::IBuffer* positionsBuffer = nullptr;
    Diligent::IBuffer* orientationsBuffer = nullptr;
    Diligent::IBuffer* scalesBuffer = nullptr;
    std::uint32_t count = 0;
};

struct GpuEntitySceneView
{
    GpuPoseBufferView poses{};
    Diligent::IBuffer* renderableMetadataBuffer = nullptr;
    Diligent::IBuffer* renderableModelMatricesBuffer = nullptr;
    Diligent::IBuffer* renderableNormalMatricesBuffer = nullptr;
    Diligent::IBuffer* renderableVisibilityFlagsBuffer = nullptr;
    Diligent::IBuffer* renderableShadowCascadeMasksBuffer = nullptr;
    std::uint32_t entityCount = 0;
    std::uint32_t renderableCount = 0;
};

struct GpuEntityPoseMappingEntry
{
    std::uint32_t sourcePoseIndex = 0;
    std::uint32_t entityPoseIndex = 0;
};

enum GpuRenderableFlags : std::uint32_t
{
    GpuRenderableFlag_None = 0u,
    GpuRenderableFlag_Opaque = 1u << 0u,
    GpuRenderableFlag_ShadowCaster = 1u << 1u,
    GpuRenderableFlag_Transparent = 1u << 2u,
    GpuRenderableFlag_UsesGpuPose = 1u << 3u,
};

struct GpuRenderableMetadata
{
    std::uint32_t entityPoseIndex = 0xffffffffu;
    std::uint32_t envIndex = 0u;
    std::uint32_t flags = GpuRenderableFlag_None;
    std::uint32_t reserved = 0u;
    Diligent::float4 localBoundsMin{};
    Diligent::float4 localBoundsMax{};
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_SCENE_H
