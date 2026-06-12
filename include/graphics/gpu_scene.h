#ifndef CRESSIM_NEO_GRAPHICS_GPU_SCENE_H
#define CRESSIM_NEO_GRAPHICS_GPU_SCENE_H

#include "common/flags.h"
#include "common/scene_primitives.h"
#include "graphics/gpu_scene_pipeline_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <array>
#include <cstdint>

namespace cressim::neo::graphics
{

enum class GpuLightType : std::uint32_t
{
    Directional = 0u,
    Point       = 1u,
    Spot        = 2u,
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

struct GpuLocalLightSelection
{
    std::uint32_t localLightCount         = 0u;
    std::uint32_t shadowedLocalLightCount = 0u;
    std::uint32_t shadowedPointLightCount = 0u;
    std::uint32_t reserved0               = 0u;
    std::array<std::uint32_t, kForwardLocalLightCap> lightIndices{};
};

enum class GpuRenderableFlags : std::uint32_t
{
    None         = 0u,
    Active       = 1u << 0u,
    Opaque       = 1u << 1u,
    ShadowCaster = 1u << 2u,
};
CRESSIM_NEO_DEFINE_ENUM_FLAGS(GpuRenderableFlags)

enum class GpuRenderableDeformableType : std::uint32_t
{
    None     = 0u,
    SoftBody = 1u,
    Curve    = 2u,
};

struct GpuRenderableMetadata
{
    std::uint32_t flags             = static_cast<std::uint32_t>(GpuRenderableFlags::None);
    std::uint32_t deformVertexBase  = 0u;
    std::uint32_t deformNormalBase  = 0u;
    std::uint32_t deformableIndex   = 0xffffffffu;
    std::uint32_t deformVertexCount = 0u;
    std::uint32_t deformableType    = 0u;
    std::uint32_t reserved1         = 0u;
    std::uint32_t reserved2         = 0u;
    Diligent::float4 localBoundsMin{};
    Diligent::float4 localBoundsMax{};
};
static_assert(sizeof(GpuRenderableMetadata) == 64u);

struct GpuSoftBodyVertexBinding
{
    std::uint32_t particleIndex = 0u;
    std::uint32_t reserved0     = 0u;
    std::uint32_t reserved1     = 0u;
    std::uint32_t reserved2     = 0u;
};

struct GpuRenderableQueueInfo
{
    std::uint32_t opaqueCommandIndex      = 0xffffffffu;
    std::uint32_t shadowCommandBaseIndex  = 0xffffffffu;
    std::uint32_t localShadowCommandIndex = 0xffffffffu;
    std::uint32_t reserved0               = 0u;
};

struct GpuEntitySceneView
{
    common::SceneLayoutDesc layout{};
    common::PoseBufferView poses{};
    Diligent::IBuffer *renderableMetadataBuffer           = nullptr;
    Diligent::IBuffer *renderableQueueInfoBuffer          = nullptr;
    Diligent::IBuffer *renderableVisibilityFlagsBuffer    = nullptr;
    Diligent::IBuffer *renderableShadowCascadeMasksBuffer = nullptr;
    Diligent::IBuffer *cameraInputsBuffer                 = nullptr;
    Diligent::IBuffer *preparedCamerasBuffer              = nullptr;
    Diligent::IBuffer *lightInputsBuffer                  = nullptr;
    Diligent::IBuffer *localLightSelectionBuffer          = nullptr;
    Diligent::IBuffer *softBodyVertexBindingBuffer        = nullptr;
    std::uint32_t entityCount                             = 0;
    std::uint32_t renderableCount                         = 0;
    std::uint32_t cameraCount                             = 0;
    std::uint32_t lightCount                              = 0;
    std::uint64_t bindingGeneration                       = 0;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GPU_SCENE_H
