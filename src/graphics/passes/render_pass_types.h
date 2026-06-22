#ifndef CRESSIM_NEO_GRAPHICS_PASSES_RENDER_PASS_TYPES_H
#define CRESSIM_NEO_GRAPHICS_PASSES_RENDER_PASS_TYPES_H

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "common/id.h"
#include "gpu/gpu_types.h"
#include "graphics/renderer.h"

#include <optional>
#include <vector>

namespace cressim::neo::graphics
{

enum class MainPassClass
{
    ForwardOpaque,
    ForwardTransparent,
};

struct ResolvedCameraView
{
    common::EntityId entityId   = common::kInvalidEntityId;
    CameraData::Product product = CameraData::Product::Color;
    gpu::GpuRenderTargetBinding outputBinding{};
    gpu::GpuRenderTargetDesc outputTargetDesc{};
    gpu::GpuRenderViewport viewport{};
    bool useOutputViewport              = false;
    bool clearColor                     = true;
    bool clearDepth                     = true;
    Diligent::float4 clearColorValue    = {0.0f, 0.0f, 0.0f, 1.0f};
    float clearDepthValue               = 1.0f;
    CameraBackgroundMode backgroundMode = CameraBackgroundMode::ClearColor;
    std::uint32_t envIndex              = 0u;
    std::uint32_t cameraSlot            = 0u;
    std::uint32_t globalCameraIndex     = 0u;
};

struct CameraBatchView
{
    gpu::GpuRenderTargetBinding renderBinding{};
    gpu::GpuRenderTargetDesc renderTargetDesc{};
    std::vector<ResolvedCameraView> cameras{};
};

struct DisplayResolveRequest
{
    RenderFrameOptions::PresentedExplicitOutput::SourceKind sourceKind =
        RenderFrameOptions::PresentedExplicitOutput::SourceKind::Color;
    gpu::GpuRenderTargetBinding sourceBinding{};
    gpu::GpuRenderTargetDesc sourceTargetDesc{};
    bool sourceIsDisplayEncoded = false;
    gpu::GpuPresentationTargetDesc presentationTarget{};
    ToneMapper toneMapper            = ToneMapper::Reinhard;
    float exposure                   = 1.0f;
    bool clearColor                  = false;
    bool clearDepth                  = false;
    bool preserveAspectRatio         = false;
    Diligent::float4 clearColorValue = {0.0f, 0.0f, 0.0f, 1.0f};
    float clearDepthValue            = 1.0f;
    float nearClip                   = 0.01f;
    float farClip                    = 1000.0f;
};

struct EnvMainLightState
{
    std::uint32_t mainLightIndex = kInvalidGpuSceneIndex;
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
    std::uint32_t opaqueDrawCalls      = 0;
    std::uint32_t transparentDrawCalls = 0;
    std::uint32_t shadowDrawCalls      = 0;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_PASSES_RENDER_PASS_TYPES_H
