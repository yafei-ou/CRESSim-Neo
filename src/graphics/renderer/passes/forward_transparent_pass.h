#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_TRANSPARENT_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_TRANSPARENT_PASS_H

#include "gpu/gpu_device.h"
#include "graphics/renderer/passes/forward_draw_types.h"

namespace cressim::neo::graphics
{

namespace detail
{

class ForwardTransparentPass
{
public:
    explicit ForwardTransparentPass(gpu::GpuDevice& device);

    bool initialize();
    bool draw(gpu::GpuRenderTargetHandle target, const ForwardDrawCommand& drawCommand);

private:
    gpu::GpuDevice& mDevice;
    bool mInitialized = false;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_TRANSPARENT_PASS_H
