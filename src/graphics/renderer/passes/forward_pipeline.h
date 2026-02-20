#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H

#include "graphics/graphics_device.h"
#include "graphics/renderer/passes/render_pass_types.h"

#include <array>
#include <memory>

namespace cressim::neo::graphics
{

class GraphicsDeviceImpl;

namespace detail
{

class ForwardOpaquePass;
class ForwardTransparentPass;
class ShadowPass;

class ForwardPipeline
{
public:
    explicit ForwardPipeline(GraphicsDeviceImpl& device);
    ~ForwardPipeline();

    bool initialize();
    bool execute(const common::FrameContext& frameContext, const FrameViewData& frameView,
                 const CameraRenderQueues& queues, ForwardPassExecutionStats& outStats);

private:
    GraphicsDeviceImpl& mDevice;
    std::unique_ptr<ForwardOpaquePass> mForwardOpaquePass;
    std::unique_ptr<ForwardTransparentPass> mForwardTransparentPass;
    std::unique_ptr<ShadowPass> mShadowPass;
    std::array<RenderTargetHandle, kShadowCascadeCount> mShadowMapTargets{};
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
