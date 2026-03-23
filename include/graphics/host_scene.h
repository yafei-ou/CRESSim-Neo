#ifndef CRESSIM_NEO_GRAPHICS_HOST_SCENE_H
#define CRESSIM_NEO_GRAPHICS_HOST_SCENE_H

#include "common/id.h"
#include "common/math_types.h"
#include "gpu/gpu_scene.h"
#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"
#include "graphics/renderer/passes/forward_draw_types.h"

#include <vector>

namespace cressim::neo::graphics
{

struct RenderableInstance
{
    common::EntityId entityId = common::kInvalidEntityId;
    std::uint32_t envIndex    = 0u;
    std::uint32_t objectSlot  = 0xffffffffu;
    bool visible              = true;
    common::Transform worldTransform{};
    MeshHandle mesh{};
    MaterialHandle material{};
};

struct CameraData
{
    common::EntityId entityId = common::kInvalidEntityId;
    std::uint32_t envIndex    = 0u;
    std::uint32_t cameraSlot  = 0xffffffffu;
    common::Transform worldTransform{};
    float verticalFovDegrees = 60.0f;
    float nearClip           = 0.01f;
    float farClip            = 1000.0f;

    // Render output and scheduling controls copied from engine::CameraComponent.
    gpu::CameraOutputBinding output{};
    std::uint32_t outputWidth  = 0;
    std::uint32_t outputHeight = 0;
    gpu::GpuRenderViewport viewport{};
    bool clearColor = true;
    bool clearDepth = true;
    Diligent::float4 clearColorValue{0.02f, 0.02f, 0.03f, 1.0f};
    float clearDepthValue = 1.0f;

    std::uint32_t renderOrder = 0;
};

struct DirectionalLightData
{
    common::EntityId entityId = common::kInvalidEntityId;
    std::uint32_t envIndex    = 0u;
    std::uint32_t lightSlot   = 0xffffffffu;
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float intensity          = 1.0f;
    float shadowDistance     = 120.0f;
    float shadowFadeDistance = 20.0f;
};

struct HostSceneView
{
    const std::vector<RenderableInstance> *renderables                  = nullptr;
    const std::vector<CameraData> *cameras                              = nullptr;
    const std::vector<DirectionalLightData> *directionalLights          = nullptr;
    const std::vector<IndirectCommandRegistryEntry> *opaqueDrawRegistry = nullptr;
    const std::vector<IndirectCommandRegistryEntry> *shadowDrawRegistry = nullptr;
    const gpu::GpuEntitySceneView *gpuEntityScene                       = nullptr;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_HOST_SCENE_H
