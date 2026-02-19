#include "graphics/renderer/passes/forward_pipeline.h"

#include "graphics/device/graphics_device_impl.h"
#include "graphics/renderer/passes/pbr_pass.h"
#include "graphics/renderer/passes/shadow_pass.h"

namespace cressim::neo::graphics::detail
{

ForwardPipeline::ForwardPipeline(GraphicsDeviceImpl& device) :
    mDevice(device)
{
}

ForwardPipeline::~ForwardPipeline()
{
    if (mDevice.isValidRenderTarget(mShadowMapTarget))
    {
        mDevice.destroyRenderTarget(mShadowMapTarget);
    }
}

bool ForwardPipeline::initialize()
{
    mPbrPass = std::make_unique<PbrPass>(mDevice);
    if (!mPbrPass->initialize())
    {
        mPbrPass.reset();
        return false;
    }

    mShadowPass = std::make_unique<ShadowPass>(mDevice);
    if (!mShadowPass->initialize())
    {
        mShadowPass.reset();
        mPbrPass.reset();
        return false;
    }

    RenderTargetDesc shadowDesc{};
    shadowDesc.width = 2048;
    shadowDesc.height = 2048;
    shadowDesc.color = false;
    shadowDesc.depth = true;
    shadowDesc.shaderReadable = true;
    shadowDesc.debugName = "CRESSimNeo.ShadowMap";
    mShadowMapTarget = mDevice.createRenderTarget(shadowDesc);
    if (!mDevice.isValidRenderTarget(mShadowMapTarget))
    {
        mShadowMapTarget = {};
    }

    mInitialized = true;
    return true;
}

bool ForwardPipeline::execute(
    const common::FrameContext& frameContext,
    const FrameViewData& frameView,
    const CameraRenderQueues& queues,
    ForwardPassExecutionStats& outStats)
{
    if (!mInitialized || mPbrPass == nullptr)
    {
        return false;
    }

    outStats = {};

    bool hasShadowMap = false;
    if (frameView.hasDirectionalLight && !queues.shadowCasters.empty() && mShadowPass != nullptr && mDevice.isValidRenderTarget(mShadowMapTarget))
    {
        mDevice.setRenderTargetViewport(mShadowMapTarget, RenderViewport{});

        RenderPassBeginDesc shadowBegin{};
        shadowBegin.clearColor = false;
        shadowBegin.clearDepth = true;
        shadowBegin.clearDepthValue = 1.0f;

        mDevice.beginRenderTarget(mShadowMapTarget, frameContext, shadowBegin);
        for (const QueuedDraw& draw : queues.shadowCasters)
        {
            if (mShadowPass->draw(mShadowMapTarget, draw.drawCommand))
            {
                ++outStats.shadowDrawCalls;
            }
        }
        mDevice.endRenderTarget(mShadowMapTarget, frameContext);
        hasShadowMap = true;
    }

    mPbrPass->setShadowMapTarget(hasShadowMap ? mShadowMapTarget : RenderTargetHandle{});

    mDevice.setRenderTargetViewport(frameView.target, frameView.viewport);
    const RenderPassBeginDesc mainBegin{};
    mDevice.beginRenderTarget(frameView.target, frameContext, mainBegin);

    for (const QueuedDraw& draw : queues.opaque)
    {
        switch (draw.drawCommand.shadingModel)
        {
        case ShadingModel::Pbr:
            if (mPbrPass->draw(frameView.target, draw.drawCommand))
            {
                ++outStats.opaqueDrawCalls;
            }
            break;
        case ShadingModel::Phong:
            // Reserved for upcoming Phong pass.
            break;
        default:
            break;
        }
    }

    // Transparent pass is still scaffolded in this milestone.
    (void)queues.transparent;
    mDevice.endRenderTarget(frameView.target, frameContext);
    return true;
}

} // namespace cressim::neo::graphics::detail
