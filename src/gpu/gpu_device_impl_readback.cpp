#include "common/logger.h"
#include "gpu/gpu_device_impl.h"
#include "gpu/gpu_render_target_system_impl.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"

#include <cstring>

namespace cressim::neo::gpu
{

namespace
{

bool supportsPresentationBackend(GpuBackend backend) noexcept
{
    return backend == GpuBackend::D3D12 || backend == GpuBackend::Vulkan;
}

} // namespace

void GpuDeviceImpl::beginFrame(const common::FrameContext &frameContext)
{
    (void)frameContext;

    if (mFrameActive)
    {
        CRESSIM_LOG_WARNING("GpuDevice::beginFrame called while a frame is already active.");
        return;
    }

    mFrameActive = true;
}

void GpuDeviceImpl::endFrame(const common::FrameContext &frameContext)
{
    if (!mFrameActive)
    {
        CRESSIM_LOG_WARNING("GpuDevice::endFrame called without a matching beginFrame.");
        return;
    }

    mFrameActive = false;

    if (mRenderTargetSystem == nullptr)
    {
        return;
    }

    bool graphicsFrameFinalizedByPresent = false;
    if (mInitialized && supportsPresentationBackend(mBackend) && mGraphicsContext != nullptr)
    {
        (void)queuePresentationReadback(frameContext);
        (void)presentPrimarySwapChain();
        graphicsFrameFinalizedByPresent = (mPrimarySwapChain != nullptr);
    }

    mRenderTargetSystem->endFrame(frameContext, !graphicsFrameFinalizedByPresent);
    if (mInitialized && supportsPresentationBackend(mBackend) && mGraphicsContext != nullptr)
    {
        processCompletedPresentationReadbacks();
    }
    if (mInitialized && supportsPresentationBackend(mBackend) && mPhysicsContext != nullptr &&
        mPhysicsContext != mGraphicsContext)
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
    if (!mInitialized || !supportsPresentationBackend(mBackend) || mGraphicsContext == nullptr)
    {
        return false;
    }

    mPrimarySwapChain->Present(mDesc.presentation.syncInterval);
    if (!mPrimarySwapChain->GetDesc().IsPrimary)
    {
        mGraphicsContext->Flush();
        mGraphicsContext->FinishFrame();
    }

    return true;
}

bool GpuDeviceImpl::queuePresentationReadback(const common::FrameContext &frameContext)
{
    if (mPrimarySwapChain == nullptr || mRenderDevice == nullptr || mGraphicsContext == nullptr ||
        mPresentationReadbackFence == nullptr || mPendingPresentationReadbackRequests.empty())
    {
        return true;
    }

    Diligent::ITextureView *backBufferRtv = mPrimarySwapChain->GetCurrentBackBufferRTV();
    if (backBufferRtv == nullptr || backBufferRtv->GetTexture() == nullptr)
    {
        return false;
    }

    Diligent::ITexture *backBufferTexture = backBufferRtv->GetTexture();
    const auto &textureDesc               = backBufferTexture->GetDesc();
    if (textureDesc.Width == 0u || textureDesc.Height == 0u)
    {
        return false;
    }

    Diligent::TextureDesc stagingDesc = textureDesc;
    const std::string stagingName     = "CRESSimNeo.PresentationReadback";
    stagingDesc.Name                  = stagingName.c_str();
    stagingDesc.Type                  = Diligent::RESOURCE_DIM_TEX_2D;
    stagingDesc.ArraySize             = 1u;
    stagingDesc.BindFlags             = Diligent::BIND_NONE;
    stagingDesc.Usage                 = Diligent::USAGE_STAGING;
    stagingDesc.CPUAccessFlags        = Diligent::CPU_ACCESS_READ;
    stagingDesc.MiscFlags             = Diligent::MISC_TEXTURE_FLAG_NONE;

    for (auto it = mPendingPresentationReadbackRequests.begin();
         it != mPendingPresentationReadbackRequests.end();)
    {
        Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
        mRenderDevice->CreateTexture(stagingDesc, nullptr, &stagingTexture);
        if (stagingTexture == nullptr)
        {
            return false;
        }

        Diligent::CopyTextureAttribs copyAttribs{
            backBufferTexture, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION, stagingTexture,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION};
        mGraphicsContext->CopyTexture(copyAttribs);

        const std::uint64_t fenceValue = mNextPresentationReadbackFenceValue++;
        mGraphicsContext->EnqueueSignal(mPresentationReadbackFence, fenceValue);

        PendingPresentationReadback readbackCopy{};
        readbackCopy.requestId      = it->first;
        readbackCopy.frameIndex     = frameContext.frameIndex;
        readbackCopy.fenceValue     = fenceValue;
        readbackCopy.width          = textureDesc.Width;
        readbackCopy.height         = textureDesc.Height;
        readbackCopy.colorFormat    = textureDesc.Format;
        readbackCopy.stagingTexture = std::move(stagingTexture);
        mPendingPresentationReadbackCopies.push_back(std::move(readbackCopy));
        it = mPendingPresentationReadbackRequests.erase(it);
    }

    return true;
}

void GpuDeviceImpl::processCompletedPresentationReadbacks()
{
    for (PendingPresentationReadback &copy : mPendingPresentationReadbackCopies)
    {
        GpuPresentationReadbackEvent event{};
        if (consumePresentationReadback(copy, event))
        {
            mCompletedPresentationReadbacks[copy.requestId] = std::move(event);
        }
    }
    mPendingPresentationReadbackCopies.clear();
}

bool GpuDeviceImpl::consumePresentationReadback(PendingPresentationReadback &copy,
                                                GpuPresentationReadbackEvent &outEvent)
{
    outEvent             = {};
    outEvent.frameIndex  = copy.frameIndex;
    outEvent.colorFormat = copy.colorFormat;

    if (copy.stagingTexture == nullptr || copy.width == 0u || copy.height == 0u)
    {
        return true;
    }

    mPresentationReadbackFence->Wait(copy.fenceValue);

    Diligent::MappedTextureSubresource mappedData{};
    mGraphicsContext->MapTextureSubresource(copy.stagingTexture, 0, 0, Diligent::MAP_READ,
                                            Diligent::MAP_FLAG_DO_NOT_WAIT, nullptr, mappedData);
    if (mappedData.pData == nullptr)
    {
        return false;
    }

    outEvent.width            = copy.width;
    outEvent.height           = copy.height;
    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(copy.colorFormat);
    outEvent.rowStrideBytes   = copy.width * formatAttribs.GetElementSize();
    outEvent.colorBytes.resize(static_cast<std::size_t>(outEvent.rowStrideBytes) *
                               static_cast<std::size_t>(outEvent.height));

    const auto *srcRows = static_cast<const std::uint8_t *>(mappedData.pData);
    auto *dstRows       = outEvent.colorBytes.data();
    for (std::uint32_t y = 0; y < outEvent.height; ++y)
    {
        std::memcpy(dstRows + static_cast<std::size_t>(y) * outEvent.rowStrideBytes,
                    srcRows +
                        static_cast<std::size_t>(y) * static_cast<std::size_t>(mappedData.Stride),
                    outEvent.rowStrideBytes);
    }

    mGraphicsContext->UnmapTextureSubresource(copy.stagingTexture, 0, 0);
    return true;
}

} // namespace cressim::neo::gpu
