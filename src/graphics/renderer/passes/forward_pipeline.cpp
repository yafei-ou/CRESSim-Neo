#include "graphics/renderer/passes/forward_pipeline.h"

#include "graphics/renderer/passes/pbr_pass.h"

namespace cressim::neo::graphics::detail
{

ForwardPipeline::ForwardPipeline(GraphicsDeviceImpl& device) :
    mDevice(device)
{
}

ForwardPipeline::~ForwardPipeline() = default;

bool ForwardPipeline::initialize()
{
    mPbrPass = std::make_unique<PbrPass>(mDevice);
    if (!mPbrPass->initialize())
    {
        mPbrPass.reset();
        return false;
    }

    mInitialized = true;
    return true;
}

bool ForwardPipeline::execute(
    RenderTargetHandle target,
    const CameraRenderQueues& queues,
    ForwardPassExecutionStats& outStats)
{
    if (!mInitialized || mPbrPass == nullptr)
    {
        return false;
    }

    outStats = {};

    // Milestone 1 scaffold: shadow pass is queued but not executed yet.
    (void)queues.shadowCasters;

    for (const QueuedDraw& draw : queues.opaque)
    {
        switch (draw.drawCommand.shadingModel)
        {
        case ShadingModel::Pbr:
            if (mPbrPass->draw(target, draw.drawCommand))
            {
                ++outStats.opaqueDrawCalls;
            }
            break;
        case ShadingModel::Phong:
            // Milestone 1: Phong pass reserved for next milestone.
            break;
        default:
            break;
        }
    }

    // Milestone 1 scaffold: transparent pass is queued but not executed yet.
    (void)queues.transparent;
    return true;
}

} // namespace cressim::neo::graphics::detail
