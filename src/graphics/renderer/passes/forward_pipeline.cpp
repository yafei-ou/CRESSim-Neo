#include "graphics/renderer/passes/forward_pipeline.h"

#include "graphics/device/graphics_device_impl.h"
#include "graphics/renderer/passes/pbr_pass.h"
#include "graphics/renderer/passes/shadow_pass.h"

#include <algorithm>
#include <string>

namespace cressim::neo::graphics::detail
{

ForwardPipeline::ForwardPipeline(GraphicsDeviceImpl& device) :
    mDevice(device)
{
}

ForwardPipeline::~ForwardPipeline()
{
    for (RenderTargetHandle target : mShadowMapTargets)
    {
        if (mDevice.isValidRenderTarget(target))
        {
            mDevice.destroyRenderTarget(target);
        }
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

    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        RenderTargetDesc shadowDesc{};
        shadowDesc.width = kShadowMapResolution;
        shadowDesc.height = kShadowMapResolution;
        shadowDesc.color = false;
        shadowDesc.depth = true;
        shadowDesc.shaderReadable = true;
        shadowDesc.debugName = "CRESSimNeo.ShadowMap.Cascade" + std::to_string(cascadeIdx);
        mShadowMapTargets[cascadeIdx] = mDevice.createRenderTarget(shadowDesc);
        if (!mDevice.isValidRenderTarget(mShadowMapTargets[cascadeIdx]))
        {
            mShadowMapTargets[cascadeIdx] = {};
        }
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

    std::array<RenderTargetHandle, kShadowCascadeCount> activeShadowMaps{};
    std::uint32_t activeShadowMapCount = 0;
    if (frameView.hasDirectionalLight && frameView.shadowCascadeCount > 0 && !queues.shadowCasters.empty() && mShadowPass != nullptr)
    {
        for (std::uint32_t cascadeIdx = 0; cascadeIdx < frameView.shadowCascadeCount; ++cascadeIdx)
        {
            const RenderTargetHandle cascadeShadowMap = mShadowMapTargets[cascadeIdx];
            if (!mDevice.isValidRenderTarget(cascadeShadowMap))
            {
                continue;
            }

            mDevice.setRenderTargetViewport(cascadeShadowMap, RenderViewport{});

            RenderPassBeginDesc shadowBegin{};
            shadowBegin.clearColor = false;
            shadowBegin.clearDepth = true;
            shadowBegin.clearDepthValue = 1.0f;

            mDevice.beginRenderTarget(cascadeShadowMap, frameContext, shadowBegin);
            for (const QueuedDraw& draw : queues.shadowCasters)
            {
                if ((draw.shadowCascadeMask & (1u << cascadeIdx)) == 0u)
                {
                    continue;
                }
                ForwardDrawCommand shadowDrawCommand = draw.drawCommand;
                shadowDrawCommand.lightViewProjectionMatrix = draw.drawCommand.lightViewProjectionMatrices[cascadeIdx];
                if (mShadowPass->draw(cascadeShadowMap, shadowDrawCommand))
                {
                    ++outStats.shadowDrawCalls;
                }
            }
            mDevice.endRenderTarget(cascadeShadowMap, frameContext);
            activeShadowMaps[cascadeIdx] = cascadeShadowMap;
            activeShadowMapCount = std::max(activeShadowMapCount, cascadeIdx + 1);
        }
    }

    mPbrPass->setShadowMapTargets(activeShadowMaps, activeShadowMapCount);

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
