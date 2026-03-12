#include "graphics/renderer/passes/forward_transparent_pass.h"

namespace cressim::neo::graphics::detail
{

ForwardTransparentPass::ForwardTransparentPass(gpu::GpuDevice& device) : mDevice(device) {}

bool ForwardTransparentPass::initialize()
{
    mInitialized = true;
    return true;
}

bool ForwardTransparentPass::draw(gpu::GpuRenderTargetHandle target,
                                  const ForwardDrawCommand& drawCommand)
{
    (void)target;
    (void)drawCommand;
    return mInitialized && false;
}

} // namespace cressim::neo::graphics::detail
