#include "graphics/renderer/passes/forward_pipeline.h"

#include "graphics/renderer/passes/forward_opaque_pass.h"
#include "graphics/renderer/passes/forward_transparent_pass.h"
#include "graphics/renderer/passes/shadow_pass.h"

#include <algorithm>
#include <string>

namespace cressim::neo::graphics::detail
{

ForwardPipeline::ForwardPipeline(gpu::GpuDevice& device) : mDevice(device) {}

ForwardPipeline::~ForwardPipeline()
{
    for (gpu::GpuRenderTargetHandle target : mShadowMapTargets)
    {
        if (mDevice.renderTargetSystem().isValidRenderTarget(target))
        {
            mDevice.renderTargetSystem().destroyRenderTarget(target);
        }
    }
}

bool ForwardPipeline::initialize()
{
    mForwardOpaquePass = std::make_unique<ForwardOpaquePass>(mDevice);
    if (!mForwardOpaquePass->initialize())
    {
        mForwardOpaquePass.reset();
        return false;
    }

    mForwardTransparentPass = std::make_unique<ForwardTransparentPass>(mDevice);
    if (!mForwardTransparentPass->initialize())
    {
        mForwardTransparentPass.reset();
        mForwardOpaquePass.reset();
        return false;
    }

    mShadowPass = std::make_unique<ShadowPass>(mDevice);
    if (!mShadowPass->initialize())
    {
        mShadowPass.reset();
        mForwardTransparentPass.reset();
        mForwardOpaquePass.reset();
        return false;
    }

    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        gpu::GpuRenderTargetDesc shadowDesc{};
        shadowDesc.width              = kShadowMapResolution;
        shadowDesc.height             = kShadowMapResolution;
        shadowDesc.color              = false;
        shadowDesc.depth              = true;
        shadowDesc.shaderReadable     = true;
        shadowDesc.debugName          = "CRESSimNeo.ShadowMap.Cascade" + std::to_string(cascadeIdx);
        mShadowMapTargets[cascadeIdx] = mDevice.renderTargetSystem().createRenderTarget(shadowDesc);
        if (!mDevice.renderTargetSystem().isValidRenderTarget(mShadowMapTargets[cascadeIdx]))
        {
            mShadowMapTargets[cascadeIdx] = {};
        }
    }

    mInitialized = true;
    return true;
}

bool ForwardPipeline::execute(const common::FrameContext& frameContext,
                              const FrameViewData& frameView,
                              const gpu::GpuEntitySceneView& sceneView,
                              const CameraRenderQueues& queues,
                              ForwardPassExecutionStats& outStats)
{
    if (!mInitialized || mForwardOpaquePass == nullptr)
    {
        return false;
    }

    outStats = {};
    mForwardOpaquePass->setGpuSceneView(sceneView);
    if (mShadowPass != nullptr)
    {
        mShadowPass->setGpuSceneView(sceneView);
    }

    std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> activeShadowMaps{};
    std::uint32_t activeShadowMapCount = 0;
    if (frameView.hasDirectionalLight && frameView.shadowCascadeCount > 0 &&
        !queues.shadowCasters.empty() && mShadowPass != nullptr)
    {
        for (std::uint32_t cascadeIdx = 0; cascadeIdx < frameView.shadowCascadeCount; ++cascadeIdx)
        {
            const gpu::GpuRenderTargetHandle cascadeShadowMap = mShadowMapTargets[cascadeIdx];
            if (!mDevice.renderTargetSystem().isValidRenderTarget(cascadeShadowMap))
            {
                continue;
            }

            mDevice.renderTargetSystem().setRenderTargetViewport(cascadeShadowMap,
                                                                 gpu::GpuRenderViewport{});

            gpu::GpuRenderPassBeginDesc shadowBegin{};
            shadowBegin.clearColor      = false;
            shadowBegin.clearDepth      = true;
            shadowBegin.clearDepthValue = 1.0f;

            mDevice.renderTargetSystem().beginRenderTarget(cascadeShadowMap, frameContext,
                                                           shadowBegin);
            for (const QueuedDraw& draw : queues.shadowCasters)
            {
                if ((draw.shadowCascadeMask & (1u << cascadeIdx)) == 0u)
                {
                    continue;
                }
                if (mShadowPass->draw(cascadeShadowMap, draw.drawCommand,
                                      frameView.lightViewProjectionMatrices[cascadeIdx],
                                      cascadeIdx))
                {
                    ++outStats.shadowDrawCalls;
                }
            }
            mDevice.renderTargetSystem().endRenderTarget(cascadeShadowMap, frameContext);
            activeShadowMaps[cascadeIdx] = cascadeShadowMap;
            activeShadowMapCount         = std::max(activeShadowMapCount, cascadeIdx + 1);
        }
    }

    mForwardOpaquePass->setShadowMapTargets(activeShadowMaps, activeShadowMapCount);
    if (!mForwardOpaquePass->beginCameraFrame(frameView))
    {
        return false;
    }

    mDevice.renderTargetSystem().setRenderTargetViewport(frameView.target, frameView.viewport);
    const gpu::GpuRenderPassBeginDesc mainBegin{};
    mDevice.renderTargetSystem().beginRenderTarget(frameView.target, frameContext, mainBegin);

    for (const QueuedDraw& draw : queues.opaque)
    {
        if (mForwardOpaquePass->draw(frameView.target, draw.drawCommand))
        {
            ++outStats.opaqueDrawCalls;
        }
    }

    if (mForwardTransparentPass != nullptr)
    {
        for (const QueuedDraw& draw : queues.transparent)
        {
            if (mForwardTransparentPass->draw(frameView.target, draw.drawCommand))
            {
                ++outStats.transparentDrawCalls;
            }
        }
    }

    mDevice.renderTargetSystem().endRenderTarget(frameView.target, frameContext);
    return true;
}

} // namespace cressim::neo::graphics::detail
