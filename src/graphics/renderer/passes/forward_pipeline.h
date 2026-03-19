#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H

#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"
#include "graphics/host_scene.h"
#include "graphics/renderer/passes/render_pass_types.h"

#include <array>
#include <memory>

namespace cressim::neo::graphics
{

namespace detail
{

class ForwardOpaquePass;
class ShadowPass;

class ForwardPipeline
{
public:
    ForwardPipeline(gpu::GpuDevice& device, RenderResourceManager& resourceManager);
    ~ForwardPipeline();

    bool initialize();
    bool execute(const common::FrameContext& frameContext, const FrameViewData& frameView,
                 const HostSceneView& sceneView, ForwardPassExecutionStats& outStats);

private:
    struct GpuIndirectState;

    gpu::GpuDevice& mDevice;
    RenderResourceManager& mResourceManager;
    std::unique_ptr<ForwardOpaquePass> mForwardOpaquePass;
    std::unique_ptr<ShadowPass> mShadowPass;
    std::unique_ptr<GpuIndirectState> mGpuIndirectState;
    std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> mShadowMapTargets{};
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
