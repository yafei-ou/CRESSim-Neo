#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H

#include "graphics/graphics_device.h"
#include "graphics/renderer/passes/forward_draw_types.h"

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
    bool draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand);

private:
    GraphicsDeviceImpl& mDevice;
    std::unique_ptr<PbrPass> mPbrPass;
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_PIPELINE_H
