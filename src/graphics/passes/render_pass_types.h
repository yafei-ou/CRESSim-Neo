#ifndef CRESSIM_NEO_GRAPHICS_PASSES_RENDER_PASS_TYPES_H
#define CRESSIM_NEO_GRAPHICS_PASSES_RENDER_PASS_TYPES_H

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "common/id.h"
#include "gpu/gpu_scene.h"
#include "gpu/gpu_types.h"

#include <optional>
#include <vector>

namespace cressim::neo::graphics
{

enum class MainPassClass
{
    ForwardOpaque,
};

struct ResolvedCameraView
{
    common::EntityId entityId = common::kInvalidEntityId;
    gpu::GpuRenderTargetBinding outputBinding{};
    gpu::GpuRenderTargetDesc outputTargetDesc{};
    gpu::GpuRenderViewport viewport{};
    bool useOutputViewport           = false;
    bool clearColor                  = true;
    bool clearDepth                  = true;
    Diligent::float4 clearColorValue = {0.02f, 0.02f, 0.03f, 1.0f};
    float clearDepthValue            = 1.0f;
    std::uint32_t envIndex           = 0u;
    std::uint32_t cameraSlot         = 0u;
    std::uint32_t globalCameraIndex  = 0u;
};

struct CameraBatchView
{
    gpu::GpuRenderTargetBinding renderBinding{};
    gpu::GpuRenderTargetDesc renderTargetDesc{};
    std::vector<ResolvedCameraView> cameras{};
};

struct DisplayResolveRequest
{
    gpu::GpuRenderTargetBinding sourceBinding{};
    gpu::GpuRenderTargetDesc sourceTargetDesc{};
    gpu::GpuPresentationTargetDesc presentationTarget{};
    bool clearColor                  = false;
    bool clearDepth                  = false;
    Diligent::float4 clearColorValue = {0.02f, 0.02f, 0.03f, 1.0f};
    float clearDepthValue            = 1.0f;
};

struct EnvMainLightState
{
    std::uint32_t mainLightIndex = gpu::kInvalidGpuSceneIndex;
    bool active                  = false;
    bool castsShadows            = false;
};

struct FrameRenderPlan
{
    std::vector<CameraBatchView> cameraBatches{};
    std::vector<EnvMainLightState> envMainLights{};
    std::optional<DisplayResolveRequest> displayResolve{};
};

struct ForwardPassExecutionStats
{
    std::uint32_t opaqueDrawCalls = 0;
    std::uint32_t shadowDrawCalls = 0;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_PASSES_RENDER_PASS_TYPES_H
