#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_TRANSPARENT_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_TRANSPARENT_PASS_H

#include "graphics/graphics_device.h"
#include "graphics/renderer/passes/forward_draw_types.h"

namespace cressim::neo::graphics
{

class GraphicsDeviceImpl;

namespace detail
{

class ForwardTransparentPass
{
public:
    explicit ForwardTransparentPass(GraphicsDeviceImpl& device);

    bool initialize();
    bool draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand);

private:
    GraphicsDeviceImpl& mDevice;
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_TRANSPARENT_PASS_H
