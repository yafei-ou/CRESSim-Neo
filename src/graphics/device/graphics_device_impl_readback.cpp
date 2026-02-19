#include "graphics/device/graphics_device_impl.h"

#include <cstring>
#include <utility>

namespace cressim::neo::graphics
{

RenderTargetReadbackRequest GraphicsDeviceImpl::requestRenderTargetReadback(RenderTargetHandle target)
{
    RenderTargetReadbackRequest request{};

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return request;
    }
    if (!it->second.desc.cpuReadback)
    {
        return request;
    }

    std::uint64_t requestId = mNextReadbackRequestId++;
    if (requestId == 0)
    {
        requestId = mNextReadbackRequestId++;
    }
    request.id = requestId;
    mPendingReadbackRequests[target.id].push_back(requestId);
    return request;
}

bool GraphicsDeviceImpl::tryGetRenderTargetReadback(RenderTargetReadbackRequest request, RenderTargetReadbackEvent& outEvent)
{
    if (request.id == 0)
    {
        return false;
    }

    const auto completedIt = mCompletedReadbacks.find(request.id);
    if (completedIt == mCompletedReadbacks.end())
    {
        return false;
    }

    outEvent = completedIt->second;
    mCompletedReadbacks.erase(completedIt);
    return true;
}

void GraphicsDeviceImpl::endFrame(const common::FrameContext& frameContext)
{
    (void)frameContext;

    if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext)
    {
        for (const PendingReadbackCopy& copy : mPendingReadbackCopies)
        {
            RenderTargetReadbackEvent event{};
            event.target = copy.target;
            event.frameIndex = copy.frameIndex;
            event.colorFormat = copy.colorFormat;
            for (const std::uint64_t requestId : copy.requestIds)
            {
                mCompletedReadbacks[requestId] = event;
            }
        }
        mPendingReadbackCopies.clear();
        return;
    }

    mImmediateContext->Flush();
    mImmediateContext->FinishFrame();

    for (const PendingReadbackCopy& copy : mPendingReadbackCopies)
    {
        RenderTargetReadbackEvent event{};
        event.target = copy.target;
        event.frameIndex = copy.frameIndex;
        event.colorFormat = copy.colorFormat;

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
                event.colorBytes.resize(static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height));

                const auto* srcRows = static_cast<const std::uint8_t*>(mappedData.pData);
                auto* dstRows = event.colorBytes.data();
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

        for (const std::uint64_t requestId : copy.requestIds)
        {
            mCompletedReadbacks[requestId] = event;
        }
    }

    mPendingReadbackCopies.clear();
}

bool GraphicsDeviceImpl::queueReadbackCopy(RenderTargetHandle target, std::uint64_t frameIndex, const std::vector<std::uint64_t>& requestIds)
{
    if (!mRenderDevice || !mImmediateContext || !mReadbackFence || mBackend != GraphicsBackend::Vulkan)
    {
        return false;
    }
    if (requestIds.empty())
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
    readbackCopy.requestIds = requestIds;
    readbackCopy.target = target;
    readbackCopy.frameIndex = frameIndex;
    readbackCopy.fenceValue = fenceValue;
    readbackCopy.width = resources.desc.width;
    readbackCopy.height = resources.desc.height;
    readbackCopy.colorFormat = resources.desc.colorFormat;
    readbackCopy.stagingTexture = std::move(stagingTexture);
    mPendingReadbackCopies.push_back(std::move(readbackCopy));
    return true;
}

} // namespace cressim::neo::graphics
