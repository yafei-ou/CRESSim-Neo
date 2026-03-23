#include "common/math_utils_runtime.h"
#include "gpu/gpu_device_impl.h"
#include "gpu/gpu_render_target_system_impl.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"

#include <algorithm>

namespace cressim::neo::gpu
{

void GpuDeviceImpl::endFrame(const common::FrameContext &frameContext)
{
    if (mRenderTargets == nullptr)
    {
        return;
    }

    if (mInitialized && mBackend == GpuBackend::Vulkan && mImmediateContext != nullptr)
    {
        (void)presentPrimarySwapChain();
    }

    mRenderTargets->endFrame(frameContext);
    if (mInitialized && mBackend == GpuBackend::Vulkan && mPhysicsContext != nullptr &&
        mPhysicsContext != mImmediateContext)
    {
        mPhysicsContext->Flush();
        mPhysicsContext->FinishFrame();
    }
}

bool GpuDeviceImpl::presentPrimarySwapChain()
{
    if (mPrimarySwapChain == nullptr)
    {
        return true;
    }
    if (!mInitialized || mBackend != GpuBackend::Vulkan || mImmediateContext == nullptr ||
        mRenderTargets == nullptr)
    {
        return false;
    }

    Diligent::ITexture *sourceTexture         = nullptr;
    const GpuRenderTargetHandle defaultTarget = mRenderTargets->defaultRenderTarget();
    if (!mRenderTargets->tryGetRenderTargetColorTexture(defaultTarget, sourceTexture) ||
        sourceTexture == nullptr)
    {
        return false;
    }

    GpuRenderTargetDesc sourceDesc{};
    if (!mRenderTargets->tryGetRenderTargetDesc(defaultTarget, sourceDesc))
    {
        return false;
    }

    const auto &swapChainDesc = mPrimarySwapChain->GetDesc();
    if (swapChainDesc.Width != sourceDesc.width || swapChainDesc.Height != sourceDesc.height)
    {
        mPrimarySwapChain->Resize(common::runtime_math::clampExtent(sourceDesc.width),
                                  common::runtime_math::clampExtent(sourceDesc.height),
                                  Diligent::SURFACE_TRANSFORM_OPTIMAL);
    }

    Diligent::ITextureView *backBufferRtv = mPrimarySwapChain->GetCurrentBackBufferRTV();
    if (backBufferRtv == nullptr || backBufferRtv->GetTexture() == nullptr)
    {
        return false;
    }
    Diligent::ITexture *backBufferTexture = backBufferRtv->GetTexture();
    if (backBufferTexture == nullptr)
    {
        return false;
    }

    const auto &srcTexDesc         = sourceTexture->GetDesc();
    const auto &dstTexDesc         = backBufferTexture->GetDesc();
    const std::uint32_t copyWidth  = std::min<std::uint32_t>(srcTexDesc.Width, dstTexDesc.Width);
    const std::uint32_t copyHeight = std::min<std::uint32_t>(srcTexDesc.Height, dstTexDesc.Height);
    if (copyWidth == 0 || copyHeight == 0)
    {
        mPrimarySwapChain->Present(mDesc.presentation.syncInterval);
        return true;
    }

    const Diligent::Box srcBox{0u, copyWidth, 0u, copyHeight, 0u, 1u};
    Diligent::CopyTextureAttribs copyAttribs{
        sourceTexture, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION, backBufferTexture,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION};
    copyAttribs.pSrcBox = &srcBox;
    copyAttribs.DstX    = 0;
    copyAttribs.DstY    = 0;
    copyAttribs.DstZ    = 0;
    mImmediateContext->CopyTexture(copyAttribs);

    mPrimarySwapChain->Present(mDesc.presentation.syncInterval);
    if (!mPrimarySwapChain->GetDesc().IsPrimary)
    {
        mImmediateContext->Flush();
        mImmediateContext->FinishFrame();
    }

    return true;
}

} // namespace cressim::neo::gpu
