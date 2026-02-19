#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H

#include "common/id.h"
#include "graphics/graphics_device.h"
#include "graphics/renderer/passes/forward_draw_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/AdvancedMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <vector>

namespace cressim::neo::graphics
{

struct FrameViewData
{
    RenderTargetHandle target{};
    RenderViewport viewport{};
    std::uint32_t outputWidth = 0;
    std::uint32_t outputHeight = 0;
    Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
    Diligent::float4x4 lightViewProjectionMatrix = Diligent::float4x4::Identity();
    Diligent::float3 cameraWorldPosition = {0.0f, 0.0f, 0.0f};
    Diligent::ViewFrustum viewFrustum{};
    bool hasDirectionalLight = false;
};

struct QueuedDraw
{
    common::EntityId entityId = common::kInvalidEntityId;
    common::ResourceId meshId = common::kInvalidResourceId;
    common::ResourceId materialId = common::kInvalidResourceId;
    float depth = 0.0f;
    bool castsShadows = true;
    bool receivesShadows = true;
    bool transparent = false;
    ForwardDrawCommand drawCommand{};
};

struct CameraRenderQueues
{
    std::vector<QueuedDraw> opaque;
    std::vector<QueuedDraw> shadowCasters;
    std::vector<QueuedDraw> transparent;
};

struct ForwardPassExecutionStats
{
    std::uint32_t opaqueDrawCalls = 0;
    std::uint32_t shadowDrawCalls = 0;
    std::uint32_t transparentDrawCalls = 0;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H
