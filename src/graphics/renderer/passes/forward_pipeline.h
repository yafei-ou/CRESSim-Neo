#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H

#include "graphics/graphics_device.h"
#include "graphics/renderer/passes/render_pass_types.h"

#include <memory>

namespace cressim::neo::graphics
{

class GraphicsDeviceImpl;

namespace detail
{

class PbrPass;

class ForwardPipeline
{
public:
    explicit ForwardPipeline(GraphicsDeviceImpl& device);
    ~ForwardPipeline();

    bool initialize();
    bool execute(
        RenderTargetHandle target,
        const CameraRenderQueues& queues,
        ForwardPassExecutionStats& outStats);

private:
    GraphicsDeviceImpl& mDevice;
    std::unique_ptr<PbrPass> mPbrPass;
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
