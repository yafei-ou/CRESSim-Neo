#include "gpu/gpu_render_target_system_impl.h"
#include "common/math_utils_runtime.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace cressim::neo::gpu
{

namespace
{

constexpr std::uint32_t kDefaultRenderTargetWidth  = 1280u;
constexpr std::uint32_t kDefaultRenderTargetHeight = 720u;

bool requiresTextureRecreate(const GpuRenderTargetDesc& currentDesc,
                             const GpuRenderTargetDesc& updatedDesc)
{
    return currentDesc.width != updatedDesc.width || currentDesc.height != updatedDesc.height ||
           currentDesc.arraySize != updatedDesc.arraySize ||
           currentDesc.layeredRendering != updatedDesc.layeredRendering ||
           currentDesc.color != updatedDesc.color || currentDesc.depth != updatedDesc.depth ||
           currentDesc.colorFormat != updatedDesc.colorFormat ||
           currentDesc.depthFormat != updatedDesc.depthFormat ||
           currentDesc.shaderReadable != updatedDesc.shaderReadable;
}

} // namespace

bool GpuRenderTargetSystemImpl::initialize(const GpuRenderTargetDesc& defaultDesc,
                                           bool isVulkanBackend,
                                           Diligent::IRenderDevice* renderDevice,
                                           Diligent::IDeviceContext* immediateContext)
{
    shutdown();

    if (renderDevice == nullptr || immediateContext == nullptr)
    {
        return false;
    }

    mRenderDevice     = renderDevice;
    mImmediateContext = immediateContext;
    mIsVulkanBackend  = isVulkanBackend;

    if (mIsVulkanBackend)
    {
        Diligent::FenceDesc readbackFenceDesc{};
        readbackFenceDesc.Name = "CRESSimNeo.ReadbackFence";
        readbackFenceDesc.Type = Diligent::FENCE_TYPE_CPU_WAIT_ONLY;
        mRenderDevice->CreateFence(readbackFenceDesc, &mReadbackFence);
        if (mReadbackFence == nullptr)
        {
            shutdown();
            return false;
        }
    }

    mDefaultRenderTargetDesc = normalizeDefaultRenderTargetDesc(defaultDesc);
    mInitialized             = true;
    mDefaultRenderTarget     = createRenderTarget(mDefaultRenderTargetDesc);
    if (!isValidRenderTarget(mDefaultRenderTarget))
    {
        shutdown();
        return false;
    }

    return true;
}

void GpuRenderTargetSystemImpl::shutdown()
{
    if (mImmediateContext != nullptr)
    {
        mImmediateContext->SetRenderTargets(0, nullptr, nullptr,
                                            Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
    }

    mInitialized                   = false;
    mIsVulkanBackend               = false;
    mHasActiveRenderTarget         = false;
    mActiveRenderTargetHasDepth    = false;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    mNextRenderTargetId            = 1;
    mNextReadbackRequestId         = 1;
    mNextReadbackFenceValue        = 1;
    mDefaultRenderTarget           = {};
    mActiveRenderTarget            = {};
    mDefaultRenderTargetDesc       = {};
    mRenderTargets.clear();
    mPendingReadbackRequests.clear();
    mPendingReadbackCopies.clear();
    mCompletedReadbacks.clear();
    mReadbackFence    = nullptr;
    mRenderDevice     = nullptr;
    mImmediateContext = nullptr;
}

void GpuRenderTargetSystemImpl::endFrame(const common::FrameContext& frameContext)
{
    (void)frameContext;

    if (!mInitialized || !mIsVulkanBackend || !mImmediateContext)
    {
        for (const PendingReadbackCopy& copy : mPendingReadbackCopies)
        {
            GpuRenderTargetReadbackEvent event{};
            event.target      = copy.target;
            event.frameIndex  = copy.frameIndex;
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
        GpuRenderTargetReadbackEvent event{};
        event.target      = copy.target;
        event.frameIndex  = copy.frameIndex;
        event.colorFormat = copy.colorFormat;

        if (copy.stagingTexture != nullptr && copy.width > 0 && copy.height > 0)
        {
            if (mReadbackFence != nullptr && copy.fenceValue > 0)
            {
                mReadbackFence->Wait(copy.fenceValue);
            }

            Diligent::MappedTextureSubresource mappedData{};
            mImmediateContext->MapTextureSubresource(copy.stagingTexture, 0, 0, Diligent::MAP_READ,
                                                     Diligent::MAP_FLAG_DO_NOT_WAIT, nullptr,
                                                     mappedData);

            if (mappedData.pData != nullptr)
            {
                event.width          = copy.width;
                event.height         = copy.height;
                event.rowStrideBytes = copy.width * 4u;
                event.colorBytes.resize(static_cast<std::size_t>(event.rowStrideBytes) *
                                        static_cast<std::size_t>(event.height));

                const auto* srcRows = static_cast<const std::uint8_t*>(mappedData.pData);
                auto* dstRows       = event.colorBytes.data();
                for (std::uint32_t y = 0; y < event.height; ++y)
                {
                    std::memcpy(dstRows + static_cast<std::size_t>(y) * event.rowStrideBytes,
                                srcRows + static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(mappedData.Stride),
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

void GpuRenderTargetSystemImpl::fillBackendContextState(GpuBackendContext& outContext) const
{
    outContext.hasActiveRenderTarget = mHasActiveRenderTarget;
    outContext.activeRenderTargetId =
        mHasActiveRenderTarget ? mActiveRenderTarget.id : common::kInvalidResourceId;
    outContext.activeRenderTargetHasDepth    = mActiveRenderTargetHasDepth;
    outContext.activeRenderTargetColorFormat = mActiveRenderTargetColorFormat;
}

GpuRenderTargetHandle GpuRenderTargetSystemImpl::createRenderTarget(const GpuRenderTargetDesc& desc)
{
    if (!mInitialized || mRenderDevice == nullptr)
    {
        return {};
    }

    RenderTargetResources resources{};
    resources.desc        = normalizeTargetDesc(desc);
    resources.viewport    = common::runtime_math::normalizeViewport(GpuRenderViewport{});
    resources.colorFormat = resources.desc.colorFormat;
    resources.depthFormat = resources.desc.depthFormat;

    if (mIsVulkanBackend && !createRenderTargetTextures(resources.desc, resources))
    {
        return {};
    }

    const common::ResourceId id = mNextRenderTargetId++;
    mRenderTargets.emplace(id, std::move(resources));
    return GpuRenderTargetHandle{id};
}

GpuRenderTargetUpdateResult GpuRenderTargetSystemImpl::resizeRenderTarget(
    GpuRenderTargetHandle target, std::uint32_t width, std::uint32_t height)
{
    if (!mInitialized)
    {
        return GpuRenderTargetUpdateResult::Failed;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return GpuRenderTargetUpdateResult::Failed;
    }

    GpuRenderTargetDesc resizedDesc = it->second.desc;
    resizedDesc.width = common::runtime_math::clampExtent(width == 0 ? resizedDesc.width : width);
    resizedDesc.height =
        common::runtime_math::clampExtent(height == 0 ? resizedDesc.height : height);
    if (resizedDesc.width == it->second.desc.width && resizedDesc.height == it->second.desc.height)
    {
        return GpuRenderTargetUpdateResult::Unchanged;
    }

    return reconfigureRenderTarget(target, resizedDesc);
}

GpuRenderTargetUpdateResult GpuRenderTargetSystemImpl::reconfigureRenderTarget(
    GpuRenderTargetHandle target, const GpuRenderTargetDesc& desc)
{
    if (!mInitialized)
    {
        return GpuRenderTargetUpdateResult::Failed;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return GpuRenderTargetUpdateResult::Failed;
    }

    GpuRenderTargetDesc updatedDesc = normalizeTargetDesc(desc);
    if (target.id == mDefaultRenderTarget.id)
    {
        updatedDesc = normalizeDefaultRenderTargetDesc(updatedDesc);
    }
    const bool recreateTextures = requiresTextureRecreate(it->second.desc, updatedDesc);

    RenderTargetResources updatedResources = it->second;
    updatedResources.desc                  = updatedDesc;
    updatedResources.colorFormat           = updatedDesc.colorFormat;
    updatedResources.depthFormat           = updatedDesc.depthFormat;

    if (mIsVulkanBackend && recreateTextures)
    {
        if (mImmediateContext != nullptr)
        {
            mImmediateContext->SetRenderTargets(0, nullptr, nullptr,
                                                Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
        }

        if (!createRenderTargetTextures(updatedDesc, updatedResources))
        {
            return GpuRenderTargetUpdateResult::Failed;
        }
    }

    it->second = std::move(updatedResources);

    if (target.id == mDefaultRenderTarget.id)
    {
        mDefaultRenderTargetDesc = it->second.desc;
    }

    if (mHasActiveRenderTarget && mActiveRenderTarget.id == target.id)
    {
        mActiveRenderTargetHasDepth    = it->second.desc.depth;
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        if (it->second.colorTexture != nullptr)
        {
            mActiveRenderTargetColorFormat = it->second.colorTexture->GetDesc().Format;
        }
    }

    return recreateTextures ? GpuRenderTargetUpdateResult::Recreated
                            : GpuRenderTargetUpdateResult::Unchanged;
}

void GpuRenderTargetSystemImpl::destroyRenderTarget(GpuRenderTargetHandle target)
{
    if (target.id == common::kInvalidResourceId || target.id == mDefaultRenderTarget.id)
    {
        return;
    }

    if (mHasActiveRenderTarget && mActiveRenderTarget.id == target.id)
    {
        mHasActiveRenderTarget      = false;
        mActiveRenderTargetHasDepth = false;
        mActiveRenderTarget         = {};
    }

    Diligent::TEXTURE_FORMAT targetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    const auto targetIt                        = mRenderTargets.find(target.id);
    if (targetIt != mRenderTargets.end())
    {
        targetColorFormat = targetIt->second.desc.colorFormat;
    }

    auto completeRequestWithEmptyResult = [&](std::uint64_t requestId)
    {
        GpuRenderTargetReadbackEvent event{};
        event.target                   = target;
        event.colorFormat              = targetColorFormat;
        mCompletedReadbacks[requestId] = std::move(event);
    };

    const auto pendingRequestsIt = mPendingReadbackRequests.find(target.id);
    if (pendingRequestsIt != mPendingReadbackRequests.end())
    {
        for (const std::uint64_t requestId : pendingRequestsIt->second)
        {
            completeRequestWithEmptyResult(requestId);
        }
        mPendingReadbackRequests.erase(pendingRequestsIt);
    }

    mPendingReadbackCopies.erase(
        std::remove_if(mPendingReadbackCopies.begin(), mPendingReadbackCopies.end(),
                       [&](const PendingReadbackCopy& copy)
                       {
                           if (copy.target.id != target.id)
                           {
                               return false;
                           }
                           for (const std::uint64_t requestId : copy.requestIds)
                           {
                               completeRequestWithEmptyResult(requestId);
                           }
                           return true;
                       }),
        mPendingReadbackCopies.end());
    mRenderTargets.erase(target.id);
}

bool GpuRenderTargetSystemImpl::isValidRenderTarget(GpuRenderTargetHandle target) const
{
    if (target.id == common::kInvalidResourceId)
    {
        return false;
    }
    return mRenderTargets.find(target.id) != mRenderTargets.end();
}

bool GpuRenderTargetSystemImpl::tryGetRenderTargetDesc(GpuRenderTargetHandle target,
                                                       GpuRenderTargetDesc& outDesc) const
{
    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        outDesc = {};
        return false;
    }

    outDesc = it->second.desc;
    return true;
}

GpuRenderTargetHandle GpuRenderTargetSystemImpl::defaultRenderTarget() const
{
    return mDefaultRenderTarget;
}

void GpuRenderTargetSystemImpl::setRenderTargetViewport(GpuRenderTargetHandle target,
                                                        const GpuRenderViewport& viewport)
{
    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return;
    }

    it->second.viewport = common::runtime_math::normalizeViewport(viewport);
}

void GpuRenderTargetSystemImpl::beginRenderTarget(GpuRenderTargetHandle target,
                                                  const common::FrameContext& frameContext,
                                                  const GpuRenderPassBeginDesc& beginDesc)
{
    (void)frameContext;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;

    if (!mInitialized || !mIsVulkanBackend || mImmediateContext == nullptr)
    {
        return;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return;
    }

    Diligent::ITextureView* colorRtv = nullptr;
    Diligent::ITextureView* depthDsv = nullptr;
    if (it->second.colorTexture != nullptr)
    {
        colorRtv = it->second.colorRenderTargetView;
        mActiveRenderTargetColorFormat = it->second.colorTexture->GetDesc().Format;
    }
    if (it->second.depthTexture != nullptr)
    {
        depthDsv = it->second.depthStencilView;
    }

    if (colorRtv == nullptr && depthDsv == nullptr)
    {
        return;
    }

    if (colorRtv != nullptr)
    {
        mImmediateContext->SetRenderTargets(1, &colorRtv, depthDsv,
                                            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        if (beginDesc.clearColor)
        {
            mImmediateContext->ClearRenderTarget(
                colorRtv, beginDesc.clearColorValue,
                Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
    }
    else
    {
        mImmediateContext->SetRenderTargets(0, nullptr, depthDsv,
                                            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    if (depthDsv != nullptr && beginDesc.clearDepth)
    {
        mImmediateContext->ClearDepthStencil(depthDsv, Diligent::CLEAR_DEPTH_FLAG,
                                             beginDesc.clearDepthValue, 0,
                                             Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    const float targetWidth          = static_cast<float>(it->second.desc.width);
    const float targetHeight         = static_cast<float>(it->second.desc.height);
    const GpuRenderViewport viewport = it->second.viewport;

    Diligent::Viewport diligentViewport{};
    diligentViewport.TopLeftX = viewport.x * targetWidth;
    diligentViewport.TopLeftY = viewport.y * targetHeight;
    diligentViewport.Width    = viewport.width * targetWidth;
    diligentViewport.Height   = viewport.height * targetHeight;
    diligentViewport.MinDepth = 0.0f;
    diligentViewport.MaxDepth = 1.0f;
    mImmediateContext->SetViewports(1, &diligentViewport, it->second.desc.width,
                                    it->second.desc.height);

    mActiveRenderTarget         = target;
    mHasActiveRenderTarget      = true;
    mActiveRenderTargetHasDepth = (depthDsv != nullptr);
}

void GpuRenderTargetSystemImpl::endRenderTarget(GpuRenderTargetHandle target,
                                                const common::FrameContext& frameContext)
{
    if (mHasActiveRenderTarget && mActiveRenderTarget.id == target.id)
    {
        mHasActiveRenderTarget         = false;
        mActiveRenderTargetHasDepth    = false;
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        mActiveRenderTarget            = {};
    }

    if (mIsVulkanBackend && mImmediateContext != nullptr)
    {
        mImmediateContext->SetRenderTargets(0, nullptr, nullptr,
                                            Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
    }

    const auto pendingRequestsIt = mPendingReadbackRequests.find(target.id);
    if (pendingRequestsIt == mPendingReadbackRequests.end() || pendingRequestsIt->second.empty())
    {
        return;
    }

    const std::vector<std::uint64_t> requestIds = pendingRequestsIt->second;
    mPendingReadbackRequests.erase(pendingRequestsIt);

    if (queueReadbackCopy(target, frameContext.frameIndex, requestIds))
    {
        return;
    }

    GpuRenderTargetReadbackEvent event{};
    event.target        = target;
    event.frameIndex    = frameContext.frameIndex;
    const auto targetIt = mRenderTargets.find(target.id);
    if (targetIt != mRenderTargets.end())
    {
        event.colorFormat = targetIt->second.desc.colorFormat;
    }
    for (const std::uint64_t requestId : requestIds)
    {
        mCompletedReadbacks[requestId] = event;
    }
}

GpuRenderTargetReadbackRequest GpuRenderTargetSystemImpl::requestRenderTargetReadback(
    GpuRenderTargetHandle target)
{
    GpuRenderTargetReadbackRequest request{};

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
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

bool GpuRenderTargetSystemImpl::tryGetRenderTargetReadback(GpuRenderTargetReadbackRequest request,
                                                           GpuRenderTargetReadbackEvent& outEvent)
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

bool GpuRenderTargetSystemImpl::tryGetRenderTargetColorTexture(GpuRenderTargetHandle target,
                                                               Diligent::ITexture*& outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || !mIsVulkanBackend)
    {
        return false;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end() || it->second.colorTexture == nullptr)
    {
        return false;
    }

    outTexture = it->second.colorTexture;
    return true;
}

bool GpuRenderTargetSystemImpl::tryGetRenderTargetDepthTexture(GpuRenderTargetHandle target,
                                                               Diligent::ITexture*& outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || !mIsVulkanBackend)
    {
        return false;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end() || it->second.depthTexture == nullptr)
    {
        return false;
    }

    outTexture = it->second.depthTexture;
    return true;
}

GpuRenderTargetDesc GpuRenderTargetSystemImpl::normalizeDefaultRenderTargetDesc(
    const GpuRenderTargetDesc& desc) const
{
    GpuRenderTargetDesc normalized = desc;
    normalized.width               = common::runtime_math::clampExtent(
        normalized.width == 0 ? kDefaultRenderTargetWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(
        normalized.height == 0 ? kDefaultRenderTargetHeight : normalized.height);
    normalized.color = true;
    normalized.arraySize = 1u;
    normalized.layeredRendering = false;
    if (normalized.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        normalized.colorFormat = Diligent::TEX_FORMAT_RGBA8_UNORM;
    }
    if (normalized.debugName.empty())
    {
        normalized.debugName = "CRESSimNeo.Default";
    }
    return normalized;
}

GpuRenderTargetDesc GpuRenderTargetSystemImpl::normalizeTargetDesc(
    const GpuRenderTargetDesc& desc) const
{
    GpuRenderTargetDesc normalized    = desc;
    const std::uint32_t fallbackWidth = common::runtime_math::clampExtent(
        mDefaultRenderTargetDesc.width == 0 ? kDefaultRenderTargetWidth
                                            : mDefaultRenderTargetDesc.width);
    const std::uint32_t fallbackHeight = common::runtime_math::clampExtent(
        mDefaultRenderTargetDesc.height == 0 ? kDefaultRenderTargetHeight
                                             : mDefaultRenderTargetDesc.height);
    normalized.width =
        common::runtime_math::clampExtent(normalized.width == 0 ? fallbackWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(
        normalized.height == 0 ? fallbackHeight : normalized.height);
    if (normalized.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        normalized.colorFormat = mDefaultRenderTargetDesc.colorFormat;
    }
    if (!normalized.color && !normalized.depth)
    {
        normalized.color = true;
    }
    normalized.arraySize        = std::max<std::uint32_t>(normalized.arraySize, 1u);
    if (normalized.debugName.empty())
    {
        normalized.debugName = "CRESSimNeo.RenderTarget";
    }
    return normalized;
}

bool GpuRenderTargetSystemImpl::createRenderTargetTextures(const GpuRenderTargetDesc& desc,
                                                           RenderTargetResources& resources)
{
    if (!mRenderDevice || (!desc.color && !desc.depth))
    {
        return false;
    }

    resources.colorTexture          = nullptr;
    resources.depthTexture          = nullptr;
    resources.colorRenderTargetView = nullptr;
    resources.depthStencilView      = nullptr;

    if (desc.color)
    {
        const Diligent::TEXTURE_FORMAT colorFormat = resources.colorFormat;
        if (colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
        {
            return false;
        }

        Diligent::TextureDesc colorDesc{};
        const std::string colorName = desc.debugName + ".Color";
        colorDesc.Name              = colorName.c_str();
        colorDesc.Type = (desc.layeredRendering || desc.arraySize > 1u)
                             ? Diligent::RESOURCE_DIM_TEX_2D_ARRAY
                             : Diligent::RESOURCE_DIM_TEX_2D;
        colorDesc.Width             = desc.width;
        colorDesc.Height            = desc.height;
        colorDesc.MipLevels         = 1;
        colorDesc.ArraySize         = desc.arraySize;
        colorDesc.Format            = colorFormat;
        colorDesc.BindFlags         = Diligent::BIND_RENDER_TARGET;
        if (desc.shaderReadable)
        {
            colorDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
        }
        colorDesc.Usage = Diligent::USAGE_DEFAULT;

        mRenderDevice->CreateTexture(colorDesc, nullptr, &resources.colorTexture);
        if (!resources.colorTexture)
        {
            return false;
        }

        if (desc.layeredRendering && colorDesc.Type == Diligent::RESOURCE_DIM_TEX_2D_ARRAY)
        {
            Diligent::TextureViewDesc rtvDesc{};
            rtvDesc.ViewType        = Diligent::TEXTURE_VIEW_RENDER_TARGET;
            rtvDesc.TextureDim      = Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
            rtvDesc.MostDetailedMip = 0u;
            rtvDesc.NumMipLevels    = 1u;
            rtvDesc.FirstArraySlice = 0u;
            rtvDesc.NumArraySlices  = colorDesc.ArraySize;
            resources.colorTexture->CreateView(rtvDesc, &resources.colorRenderTargetView);
        }
        else
        {
            resources.colorRenderTargetView =
                resources.colorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET);
        }

        if (resources.colorRenderTargetView == nullptr)
        {
            return false;
        }
    }

    if (desc.depth)
    {
        const Diligent::TEXTURE_FORMAT depthFormat = resources.depthFormat;
        if (depthFormat == Diligent::TEX_FORMAT_UNKNOWN)
        {
            return false;
        }

        Diligent::TextureDesc depthDesc{};
        const std::string depthName = desc.debugName + ".Depth";
        depthDesc.Name              = depthName.c_str();
        depthDesc.Type = (desc.layeredRendering || desc.arraySize > 1u)
                             ? Diligent::RESOURCE_DIM_TEX_2D_ARRAY
                             : Diligent::RESOURCE_DIM_TEX_2D;
        depthDesc.Width             = desc.width;
        depthDesc.Height            = desc.height;
        depthDesc.MipLevels         = 1;
        depthDesc.ArraySize         = desc.arraySize;
        depthDesc.Format            = depthFormat;
        depthDesc.BindFlags         = Diligent::BIND_DEPTH_STENCIL;
        if (desc.shaderReadable)
        {
            depthDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
        }
        depthDesc.Usage = Diligent::USAGE_DEFAULT;

        mRenderDevice->CreateTexture(depthDesc, nullptr, &resources.depthTexture);
        if (!resources.depthTexture)
        {
            return false;
        }

        if (desc.layeredRendering && depthDesc.Type == Diligent::RESOURCE_DIM_TEX_2D_ARRAY)
        {
            Diligent::TextureViewDesc dsvDesc{};
            dsvDesc.ViewType        = Diligent::TEXTURE_VIEW_DEPTH_STENCIL;
            dsvDesc.TextureDim      = Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
            dsvDesc.MostDetailedMip = 0u;
            dsvDesc.NumMipLevels    = 1u;
            dsvDesc.FirstArraySlice = 0u;
            dsvDesc.NumArraySlices  = depthDesc.ArraySize;
            resources.depthTexture->CreateView(dsvDesc, &resources.depthStencilView);
        }
        else
        {
            resources.depthStencilView =
                resources.depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL);
        }

        if (resources.depthStencilView == nullptr)
        {
            return false;
        }
    }

    return true;
}

bool GpuRenderTargetSystemImpl::queueReadbackCopy(GpuRenderTargetHandle target,
                                                  std::uint64_t frameIndex,
                                                  const std::vector<std::uint64_t>& requestIds)
{
    if (!mRenderDevice || !mImmediateContext || !mReadbackFence || !mIsVulkanBackend)
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
    if (resources.colorTexture == nullptr)
    {
        return false;
    }

    Diligent::TextureDesc stagingDesc = resources.colorTexture->GetDesc();
    const std::string stagingName     = resources.desc.debugName + ".Readback";
    stagingDesc.Name                  = stagingName.c_str();
    stagingDesc.BindFlags             = Diligent::BIND_NONE;
    stagingDesc.Usage                 = Diligent::USAGE_STAGING;
    stagingDesc.CPUAccessFlags        = Diligent::CPU_ACCESS_READ;
    stagingDesc.MiscFlags             = Diligent::MISC_TEXTURE_FLAG_NONE;

    Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
    mRenderDevice->CreateTexture(stagingDesc, nullptr, &stagingTexture);
    if (stagingTexture == nullptr)
    {
        return false;
    }

    Diligent::CopyTextureAttribs copyAttribs{
        resources.colorTexture, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION, stagingTexture,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION};
    mImmediateContext->CopyTexture(copyAttribs);

    const std::uint64_t fenceValue = mNextReadbackFenceValue++;
    mImmediateContext->EnqueueSignal(mReadbackFence, fenceValue);

    PendingReadbackCopy readbackCopy{};
    readbackCopy.requestIds     = requestIds;
    readbackCopy.target         = target;
    readbackCopy.frameIndex     = frameIndex;
    readbackCopy.fenceValue     = fenceValue;
    readbackCopy.width          = resources.desc.width;
    readbackCopy.height         = resources.desc.height;
    readbackCopy.colorFormat    = resources.desc.colorFormat;
    readbackCopy.stagingTexture = std::move(stagingTexture);
    mPendingReadbackCopies.push_back(std::move(readbackCopy));
    return true;
}

} // namespace cressim::neo::gpu
