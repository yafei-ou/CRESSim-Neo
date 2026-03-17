#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H

#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"
#include "graphics/renderer/passes/render_pass_types.h"

#include <array>
#include <memory>

namespace cressim::neo::graphics
{

namespace detail
{

class ForwardOpaquePass;
class ForwardTransparentPass;
class ShadowPass;

class ForwardPipeline
{
public:
    explicit ForwardPipeline(gpu::GpuDevice& device);
    ~ForwardPipeline();

    bool initialize();
    bool execute(const common::FrameContext& frameContext, const FrameViewData& frameView,
                 const gpu::GpuEntitySceneView& sceneView, const CameraRenderQueues& queues,
                 ForwardPassExecutionStats& outStats);

private:
    gpu::GpuDevice& mDevice;
    std::unique_ptr<ForwardOpaquePass> mForwardOpaquePass;
    std::unique_ptr<ForwardTransparentPass> mForwardTransparentPass;
    std::unique_ptr<ShadowPass> mShadowPass;
    std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> mShadowMapTargets{};
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
