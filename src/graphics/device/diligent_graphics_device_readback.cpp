#include "graphics/device/diligent_graphics_device.h"

#include <cstring>
#include <utility>

namespace cressim::neo::graphics
{

void DiligentGraphicsDevice::requestReadback(RenderTargetHandle target)
{
    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return;
    }
    if (!it->second.desc.cpuReadback)
    {
        return;
    }

    // Deduplicates repeated requests in the same frame.
    mPendingReadbacks.insert(target.id);
}

bool DiligentGraphicsDevice::tryPopReadbackEvent(RenderTargetReadbackEvent& outEvent)
{
    if (mCompletedReadbacks.empty())
    {
        return false;
    }

    outEvent = mCompletedReadbacks.front();
    mCompletedReadbacks.pop_front();
    return true;
}

void DiligentGraphicsDevice::endFrame(const common::FrameContext& frameContext)
{
    (void)frameContext;

    if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext)
    {
        // Requests that did not get rendered this frame are discarded.
        // TODO: keep/age these requests if later behavior needs cross-frame persistence.
        mPendingReadbacks.clear();
        mPendingReadbackCopies.clear();
        return;
    }

    bool frameFinishedBySwapChain = false;
    if (mSwapChain != nullptr)
    {
        // For Vulkan primary swap chains, Present() internally flushes and finishes the frame.
        mSwapChain->Present(mDesc.debugViewer.syncInterval);
        frameFinishedBySwapChain = mSwapChain->GetDesc().IsPrimary;
    }

    if (!frameFinishedBySwapChain)
    {
        mImmediateContext->Flush();
        mImmediateContext->FinishFrame();
    }

    for (const PendingReadbackCopy& copy : mPendingReadbackCopies)
    {
        RenderTargetReadbackEvent event{};
        event.target = copy.target;
        event.frameIndex = copy.frameIndex;

        if (copy.stagingTexture != nullptr && copy.width > 0 && copy.height > 0)
        {
            if (mReadbackFence != nullptr && copy.fenceValue > 0)
            {
                mReadbackFence->Wait(copy.fenceValue);
            }

            Diligent::MappedTextureSubresource mappedData{};
            mImmediateContext->MapTextureSubresource(
                copy.stagingTexture,
                0,
                0,
                Diligent::MAP_READ,
                Diligent::MAP_FLAG_DO_NOT_WAIT,
                nullptr,
                mappedData);

            if (mappedData.pData != nullptr)
            {
                event.width = copy.width;
                event.height = copy.height;
                event.rowStrideBytes = copy.width * 4u;
                event.colorRgba8.resize(static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height));

                const auto* srcRows = static_cast<const std::uint8_t*>(mappedData.pData);
                auto* dstRows = event.colorRgba8.data();
                for (std::uint32_t y = 0; y < event.height; ++y)
                {
                    std::memcpy(
                        dstRows + static_cast<std::size_t>(y) * event.rowStrideBytes,
                        srcRows + static_cast<std::size_t>(y) * static_cast<std::size_t>(mappedData.Stride),
                        event.rowStrideBytes);
                }

                mImmediateContext->UnmapTextureSubresource(copy.stagingTexture, 0, 0);
            }
        }

        mCompletedReadbacks.push_back(std::move(event));
    }

    mPendingReadbacks.clear();
    mPendingReadbackCopies.clear();
}

bool DiligentGraphicsDevice::queueReadbackCopy(RenderTargetHandle target, std::uint64_t frameIndex)
{
    if (!mRenderDevice || !mImmediateContext || !mReadbackFence || mBackend != GraphicsBackend::Vulkan)
    {
        return false;
    }

    const auto targetIt = mRenderTargets.find(target.id);
    if (targetIt == mRenderTargets.end())
    {
        return false;
    }

    const RenderTargetResources& resources = targetIt->second;
    if (!resources.desc.cpuReadback || resources.colorTexture == nullptr)
    {
        return false;
    }

    Diligent::TextureDesc stagingDesc = resources.colorTexture->GetDesc();
    const std::string stagingName = resources.desc.debugName + ".Readback";
    stagingDesc.Name = stagingName.c_str();
    stagingDesc.BindFlags = Diligent::BIND_NONE;
    stagingDesc.Usage = Diligent::USAGE_STAGING;
    stagingDesc.CPUAccessFlags = Diligent::CPU_ACCESS_READ;
    stagingDesc.MiscFlags = Diligent::MISC_TEXTURE_FLAG_NONE;

    Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
    mRenderDevice->CreateTexture(stagingDesc, nullptr, &stagingTexture);
    if (stagingTexture == nullptr)
    {
        return false;
    }

    Diligent::CopyTextureAttribs copyAttribs{
        resources.colorTexture,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        stagingTexture,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION};
    mImmediateContext->CopyTexture(copyAttribs);

    const std::uint64_t fenceValue = mNextReadbackFenceValue++;
    mImmediateContext->EnqueueSignal(mReadbackFence, fenceValue);

    PendingReadbackCopy readbackCopy{};
    readbackCopy.target = target;
    readbackCopy.frameIndex = frameIndex;
    readbackCopy.fenceValue = fenceValue;
    readbackCopy.width = resources.desc.width;
    readbackCopy.height = resources.desc.height;
    readbackCopy.stagingTexture = std::move(stagingTexture);
    mPendingReadbackCopies.push_back(std::move(readbackCopy));
    return true;
}

} // namespace cressim::neo::graphics
