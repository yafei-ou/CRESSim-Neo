#include "gpu/gpu_render_target_system_impl.h"
#include "common/math_utils_runtime.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

namespace cressim::neo::gpu
{

namespace
{

constexpr std::uint32_t kDefaultRenderTargetWidth  = 1280u;
constexpr std::uint32_t kDefaultRenderTargetHeight = 720u;

bool requiresTextureRecreate(const GpuRenderTargetDesc &currentDesc,
                             const GpuRenderTargetDesc &updatedDesc)
{
    return currentDesc.width != updatedDesc.width || currentDesc.height != updatedDesc.height ||
           currentDesc.arraySize != updatedDesc.arraySize ||
           currentDesc.layeredRendering != updatedDesc.layeredRendering ||
           currentDesc.color != updatedDesc.color || currentDesc.depth != updatedDesc.depth ||
           currentDesc.colorFormat != updatedDesc.colorFormat ||
           currentDesc.depthFormat != updatedDesc.depthFormat ||
           currentDesc.shaderReadable != updatedDesc.shaderReadable;
}

bool supportsDiligentRenderTargets(GpuBackend backend) noexcept
{
    return backend == GpuBackend::D3D12 || backend == GpuBackend::Vulkan;
}

} // namespace

GpuRenderTargetDesc normalizeDefaultRenderTargetDesc(const GpuRenderTargetDesc &desc)
{
    GpuRenderTargetDesc normalized = desc;
    normalized.width               = common::runtime_math::clampExtent(
        normalized.width == 0 ? kDefaultRenderTargetWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(
        normalized.height == 0 ? kDefaultRenderTargetHeight : normalized.height);
    if (normalized.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        normalized.colorFormat = Diligent::TEX_FORMAT_RGBA16_FLOAT;
    }
    if (normalized.depth && normalized.depthFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        normalized.depthFormat = Diligent::TEX_FORMAT_D32_FLOAT;
    }
    if (!normalized.color && !normalized.depth)
    {
        normalized.color = true;
    }
    normalized.arraySize = std::max<std::uint32_t>(normalized.arraySize, 1u);
    return normalized;
}

std::uint64_t GpuRenderTargetSystemImpl::bindingKey(const GpuRenderTargetBinding &binding) noexcept
{
    return (static_cast<std::uint64_t>(binding.firstLayer) << 32u) |
           static_cast<std::uint64_t>(std::max(binding.layerCount, 1u));
}

GpuRenderTargetBinding GpuRenderTargetSystemImpl::normalizeBinding(
    const GpuRenderTargetBinding &binding, const RenderTargetResources &resources) const
{
    GpuRenderTargetBinding normalized = binding;
    normalized.target                 = GpuRenderTargetHandle{binding.target.id};
    normalized.firstLayer = std::min(normalized.firstLayer, resources.desc.arraySize - 1u);
    normalized.layerCount = std::max(normalized.layerCount, 1u);
    normalized.layerCount =
        std::min(normalized.layerCount, resources.desc.arraySize - normalized.firstLayer);
    return normalized;
}

bool GpuRenderTargetSystemImpl::initialize(GpuBackend backend,
                                           Diligent::IRenderDevice *renderDevice,
                                           Diligent::IDeviceContext *graphicsContext)
{
    shutdown();

    mBackend                          = backend;
    const bool requiresGraphicsDevice = supportsDiligentRenderTargets(mBackend);
    if (requiresGraphicsDevice && (renderDevice == nullptr || graphicsContext == nullptr))
    {
        return false;
    }

    mRenderDevice    = renderDevice;
    mGraphicsContext = graphicsContext;

    if (supportsDiligentRenderTargets(mBackend))
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

    mInitialized = true;
    return true;
}

void GpuRenderTargetSystemImpl::shutdown()
{
    if (mGraphicsContext != nullptr)
    {
        mGraphicsContext->SetRenderTargets(0, nullptr, nullptr,
                                           Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
    }

    mInitialized                   = false;
    mBackend                       = GpuBackend::Null;
    mHasActiveRenderTarget         = false;
    mActiveRenderTargetHasDepth    = false;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    mNextRenderTargetId            = 1;
    mNextReadbackRequestId         = 1;
    mNextReadbackFenceValue        = 1;
    mActiveRenderTargetBinding     = {};
    mRenderTargets.clear();
    mPendingReadbackRequests.clear();
    mPendingReadbackCopies.clear();
    mCompletedReadbacks.clear();
    mReadbackFence   = nullptr;
    mRenderDevice    = nullptr;
    mGraphicsContext = nullptr;
}

void GpuRenderTargetSystemImpl::endFrame(const common::FrameContext &frameContext)
{
    (void)frameContext;

    if (!mInitialized || !supportsDiligentRenderTargets(mBackend) || !mGraphicsContext)
    {
        for (const PendingReadbackCopy &copy : mPendingReadbackCopies)
        {
            GpuRenderTargetReadbackEvent event{};
            event.binding                       = copy.binding;
            event.frameIndex                    = copy.frameIndex;
            event.colorFormat                   = copy.colorFormat;
            mCompletedReadbacks[copy.requestId] = event;
        }
        mPendingReadbackCopies.clear();
        return;
    }

    mGraphicsContext->Flush();
    mGraphicsContext->FinishFrame();

    for (const PendingReadbackCopy &copy : mPendingReadbackCopies)
    {
        GpuRenderTargetReadbackEvent event{};
        event.binding     = copy.binding;
        event.frameIndex  = copy.frameIndex;
        event.colorFormat = copy.colorFormat;

        if (copy.stagingTexture != nullptr && copy.width > 0 && copy.height > 0)
        {
            if (mReadbackFence != nullptr && copy.fenceValue > 0)
            {
                mReadbackFence->Wait(copy.fenceValue);
            }

            Diligent::MappedTextureSubresource mappedData{};
            mGraphicsContext->MapTextureSubresource(copy.stagingTexture, 0, 0, Diligent::MAP_READ,
                                                    Diligent::MAP_FLAG_DO_NOT_WAIT, nullptr,
                                                    mappedData);

            if (mappedData.pData != nullptr)
            {
                const auto &formatAttribs = Diligent::GetTextureFormatAttribs(copy.colorFormat);
                const std::uint32_t pixelStrideBytes = formatAttribs.GetElementSize();
                if (pixelStrideBytes == 0u)
                {
                    mGraphicsContext->UnmapTextureSubresource(copy.stagingTexture, 0, 0);
                    mCompletedReadbacks[copy.requestId] = event;
                    continue;
                }

                event.width          = copy.width;
                event.height         = copy.height;
                event.rowStrideBytes = copy.width * pixelStrideBytes;
                event.colorBytes.resize(static_cast<std::size_t>(event.rowStrideBytes) *
                                        static_cast<std::size_t>(event.height));

                const auto *srcRows = static_cast<const std::uint8_t *>(mappedData.pData);
                auto *dstRows       = event.colorBytes.data();
                for (std::uint32_t y = 0; y < event.height; ++y)
                {
                    std::memcpy(dstRows + static_cast<std::size_t>(y) * event.rowStrideBytes,
                                srcRows + static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(mappedData.Stride),
                                event.rowStrideBytes);
                }

                mGraphicsContext->UnmapTextureSubresource(copy.stagingTexture, 0, 0);
            }
        }

        mCompletedReadbacks[copy.requestId] = event;
    }

    mPendingReadbackCopies.clear();
}

void GpuRenderTargetSystemImpl::fillBackendContextState(GpuGraphicsBackendContext &outContext) const
{
    outContext.hasActiveRenderTarget = mHasActiveRenderTarget;
    outContext.activeRenderTargetBinding =
        mHasActiveRenderTarget ? mActiveRenderTargetBinding : GpuRenderTargetBinding{};
    outContext.activeRenderTargetHasDepth    = mActiveRenderTargetHasDepth;
    outContext.activeRenderTargetColorFormat = mActiveRenderTargetColorFormat;
}

GpuRenderTargetHandle GpuRenderTargetSystemImpl::createRenderTarget(const GpuRenderTargetDesc &desc)
{
    if (!mInitialized || mRenderDevice == nullptr)
    {
        return {};
    }

    RenderTargetResources resources{};
    resources.desc        = normalizeTargetDesc(desc);
    resources.colorFormat = resources.desc.colorFormat;
    resources.depthFormat = resources.desc.depthFormat;

    if (supportsDiligentRenderTargets(mBackend) &&
        !createRenderTargetTextures(resources.desc, resources))
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
    GpuRenderTargetHandle target, const GpuRenderTargetDesc &desc)
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
    const bool recreateTextures     = requiresTextureRecreate(it->second.desc, updatedDesc);

    RenderTargetResources updatedResources = it->second;
    updatedResources.desc                  = updatedDesc;
    updatedResources.colorFormat           = updatedDesc.colorFormat;
    updatedResources.depthFormat           = updatedDesc.depthFormat;

    if (supportsDiligentRenderTargets(mBackend) && recreateTextures)
    {
        if (mGraphicsContext != nullptr)
        {
            mGraphicsContext->SetRenderTargets(0, nullptr, nullptr,
                                               Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
        }

        if (!createRenderTargetTextures(updatedDesc, updatedResources))
        {
            return GpuRenderTargetUpdateResult::Failed;
        }
    }

    it->second = std::move(updatedResources);

    if (mHasActiveRenderTarget && mActiveRenderTargetBinding.target.id == target.id)
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
    if (target.id == common::kInvalidResourceId)
    {
        return;
    }

    if (mHasActiveRenderTarget && mActiveRenderTargetBinding.target.id == target.id)
    {
        mHasActiveRenderTarget      = false;
        mActiveRenderTargetHasDepth = false;
        mActiveRenderTargetBinding  = {};
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
        event.binding                  = GpuRenderTargetBinding{target, 0u, 1u};
        event.colorFormat              = targetColorFormat;
        mCompletedReadbacks[requestId] = std::move(event);
    };

    for (auto it = mPendingReadbackRequests.begin(); it != mPendingReadbackRequests.end();)
    {
        if (it->second.target.id != target.id)
        {
            ++it;
            continue;
        }

        completeRequestWithEmptyResult(it->first);
        it = mPendingReadbackRequests.erase(it);
    }

    mPendingReadbackCopies.erase(std::remove_if(mPendingReadbackCopies.begin(),
                                                mPendingReadbackCopies.end(),
                                                [&](const PendingReadbackCopy &copy)
                                                {
                                                    if (copy.binding.target.id != target.id)
                                                    {
                                                        return false;
                                                    }
                                                    completeRequestWithEmptyResult(copy.requestId);
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
                                                       GpuRenderTargetDesc &outDesc) const
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

Diligent::ITextureView *GpuRenderTargetSystemImpl::getOrCreateRenderTargetView(
    RenderTargetResources &resources, const GpuRenderTargetBinding &binding)
{
    if (resources.colorTexture == nullptr)
    {
        return nullptr;
    }

    const GpuRenderTargetBinding normalized = normalizeBinding(binding, resources);
    const std::uint64_t key                 = bindingKey(normalized);
    const auto existingIt                   = resources.colorRenderTargetViews.find(key);
    if (existingIt != resources.colorRenderTargetViews.end())
    {
        return existingIt->second;
    }

    Diligent::TextureViewDesc viewDesc{};
    viewDesc.ViewType        = Diligent::TEXTURE_VIEW_RENDER_TARGET;
    viewDesc.TextureDim      = resources.colorTexture->GetDesc().Type;
    viewDesc.MostDetailedMip = 0u;
    viewDesc.NumMipLevels    = 1u;
    viewDesc.FirstArraySlice = normalized.firstLayer;
    viewDesc.NumArraySlices  = normalized.layerCount;

    Diligent::RefCntAutoPtr<Diligent::ITextureView> view;
    resources.colorTexture->CreateView(viewDesc, &view);
    if (view == nullptr)
    {
        return nullptr;
    }

    resources.colorRenderTargetViews.emplace(key, view);
    return view;
}

Diligent::ITextureView *GpuRenderTargetSystemImpl::getOrCreateDepthStencilView(
    RenderTargetResources &resources, const GpuRenderTargetBinding &binding)
{
    if (resources.depthTexture == nullptr)
    {
        return nullptr;
    }

    const GpuRenderTargetBinding normalized = normalizeBinding(binding, resources);
    const std::uint64_t key                 = bindingKey(normalized);
    const auto existingIt                   = resources.depthStencilViews.find(key);
    if (existingIt != resources.depthStencilViews.end())
    {
        return existingIt->second;
    }

    Diligent::TextureViewDesc viewDesc{};
    viewDesc.ViewType        = Diligent::TEXTURE_VIEW_DEPTH_STENCIL;
    viewDesc.TextureDim      = resources.depthTexture->GetDesc().Type;
    viewDesc.MostDetailedMip = 0u;
    viewDesc.NumMipLevels    = 1u;
    viewDesc.FirstArraySlice = normalized.firstLayer;
    viewDesc.NumArraySlices  = normalized.layerCount;

    Diligent::RefCntAutoPtr<Diligent::ITextureView> view;
    resources.depthTexture->CreateView(viewDesc, &view);
    if (view == nullptr)
    {
        return nullptr;
    }

    resources.depthStencilViews.emplace(key, view);
    return view;
}

void GpuRenderTargetSystemImpl::setRenderTargetViewport(const GpuRenderTargetBinding &binding,
                                                        const GpuRenderViewport &viewport)
{
    const auto it = mRenderTargets.find(binding.target.id);
    if (it == mRenderTargets.end())
    {
        return;
    }

    const GpuRenderTargetBinding normalized = normalizeBinding(binding, it->second);
    it->second.viewports[bindingKey(normalized)] =
        common::runtime_math::normalizeViewport(viewport);
}

void GpuRenderTargetSystemImpl::beginRenderTarget(const GpuRenderTargetBinding &binding,
                                                  const common::FrameContext &frameContext,
                                                  const GpuRenderPassBeginDesc &beginDesc)
{
    (void)frameContext;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;

    if (!mInitialized || !supportsDiligentRenderTargets(mBackend) || mGraphicsContext == nullptr)
    {
        return;
    }

    auto it = mRenderTargets.find(binding.target.id);
    if (it == mRenderTargets.end())
    {
        return;
    }

    RenderTargetResources &resources        = it->second;
    const GpuRenderTargetBinding normalized = normalizeBinding(binding, resources);

    Diligent::ITextureView *colorRtv = nullptr;
    Diligent::ITextureView *depthDsv = nullptr;
    if (resources.colorTexture != nullptr)
    {
        colorRtv                       = getOrCreateRenderTargetView(resources, normalized);
        mActiveRenderTargetColorFormat = resources.colorTexture->GetDesc().Format;
    }
    if (resources.depthTexture != nullptr)
    {
        depthDsv = getOrCreateDepthStencilView(resources, normalized);
    }

    if (colorRtv == nullptr && depthDsv == nullptr)
    {
        return;
    }

    if (colorRtv != nullptr)
    {
        mGraphicsContext->SetRenderTargets(1, &colorRtv, depthDsv,
                                           Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        if (beginDesc.clearColor)
        {
            mGraphicsContext->ClearRenderTarget(
                colorRtv, beginDesc.clearColorValue,
                Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
    }
    else
    {
        mGraphicsContext->SetRenderTargets(0, nullptr, depthDsv,
                                           Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    if (depthDsv != nullptr && beginDesc.clearDepth)
    {
        mGraphicsContext->ClearDepthStencil(depthDsv, Diligent::CLEAR_DEPTH_FLAG,
                                            beginDesc.clearDepthValue, 0,
                                            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    const float targetWidth  = static_cast<float>(resources.desc.width);
    const float targetHeight = static_cast<float>(resources.desc.height);
    const auto viewportIt    = resources.viewports.find(bindingKey(normalized));
    const GpuRenderViewport viewport =
        viewportIt != resources.viewports.end() ? viewportIt->second : GpuRenderViewport{};

    Diligent::Viewport diligentViewport{};
    diligentViewport.TopLeftX = viewport.x * targetWidth;
    diligentViewport.TopLeftY = viewport.y * targetHeight;
    diligentViewport.Width    = viewport.width * targetWidth;
    diligentViewport.Height   = viewport.height * targetHeight;
    diligentViewport.MinDepth = 0.0f;
    diligentViewport.MaxDepth = 1.0f;
    mGraphicsContext->SetViewports(1, &diligentViewport, resources.desc.width,
                                   resources.desc.height);

    mActiveRenderTargetBinding  = normalized;
    mHasActiveRenderTarget      = true;
    mActiveRenderTargetHasDepth = (depthDsv != nullptr);
}

void GpuRenderTargetSystemImpl::endRenderTarget(const GpuRenderTargetBinding &binding,
                                                const common::FrameContext &frameContext)
{
    GpuRenderTargetBinding normalized = binding;
    const auto targetItForBinding     = mRenderTargets.find(binding.target.id);
    if (targetItForBinding != mRenderTargets.end())
    {
        normalized = normalizeBinding(binding, targetItForBinding->second);
    }

    if (mHasActiveRenderTarget && mActiveRenderTargetBinding == normalized)
    {
        mHasActiveRenderTarget         = false;
        mActiveRenderTargetHasDepth    = false;
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        mActiveRenderTargetBinding     = {};
    }

    if (supportsDiligentRenderTargets(mBackend) && mGraphicsContext != nullptr)
    {
        mGraphicsContext->SetRenderTargets(0, nullptr, nullptr,
                                           Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
    }

    std::vector<std::uint64_t> completedRequests;
    for (const auto &[requestId, requestBinding] : mPendingReadbackRequests)
    {
        if (requestBinding.target.id == normalized.target.id)
        {
            completedRequests.push_back(requestId);
        }
    }

    for (const std::uint64_t requestId : completedRequests)
    {
        const GpuRenderTargetBinding requestBinding = mPendingReadbackRequests[requestId];
        mPendingReadbackRequests.erase(requestId);
        if (queueReadbackCopy(requestBinding, frameContext.frameIndex, requestId))
        {
            continue;
        }

        GpuRenderTargetReadbackEvent event{};
        event.binding       = requestBinding;
        event.frameIndex    = frameContext.frameIndex;
        const auto targetIt = mRenderTargets.find(requestBinding.target.id);
        if (targetIt != mRenderTargets.end())
        {
            event.colorFormat = targetIt->second.desc.colorFormat;
        }
        mCompletedReadbacks[requestId] = event;
    }
}

GpuRenderTargetReadbackRequest GpuRenderTargetSystemImpl::requestRenderTargetReadback(
    const GpuRenderTargetBinding &binding)
{
    GpuRenderTargetReadbackRequest request{};

    const auto it = mRenderTargets.find(binding.target.id);
    if (it == mRenderTargets.end())
    {
        return request;
    }

    const GpuRenderTargetBinding normalized = normalizeBinding(binding, it->second);
    if (normalized.layerCount != 1u)
    {
        return request;
    }

    std::uint64_t requestId = mNextReadbackRequestId++;
    if (requestId == 0)
    {
        requestId = mNextReadbackRequestId++;
    }
    request.id = requestId;
    mPendingReadbackRequests.emplace(requestId, normalized);
    return request;
}

bool GpuRenderTargetSystemImpl::tryGetRenderTargetReadback(GpuRenderTargetReadbackRequest request,
                                                           GpuRenderTargetReadbackEvent &outEvent)
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
                                                               Diligent::ITexture *&outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || !supportsDiligentRenderTargets(mBackend))
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
                                                               Diligent::ITexture *&outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || !supportsDiligentRenderTargets(mBackend))
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

GpuRenderTargetDesc GpuRenderTargetSystemImpl::normalizeTargetDesc(
    const GpuRenderTargetDesc &desc) const
{
    GpuRenderTargetDesc normalized = normalizeDefaultRenderTargetDesc(desc);
    if (normalized.debugName.empty())
    {
        normalized.debugName = "CRESSimNeo.RenderTarget";
    }
    return normalized;
}

bool GpuRenderTargetSystemImpl::createRenderTargetTextures(const GpuRenderTargetDesc &desc,
                                                           RenderTargetResources &resources)
{
    if (!mRenderDevice || (!desc.color && !desc.depth))
    {
        return false;
    }

    resources.colorTexture = nullptr;
    resources.depthTexture = nullptr;
    resources.colorRenderTargetViews.clear();
    resources.depthStencilViews.clear();
    resources.viewports.clear();

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
        colorDesc.Type              = (desc.layeredRendering || desc.arraySize > 1u)
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

        if (getOrCreateRenderTargetView(
                resources, GpuRenderTargetBinding{{}, 0u, colorDesc.ArraySize}) == nullptr)
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
        depthDesc.Type              = (desc.layeredRendering || desc.arraySize > 1u)
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

        if (getOrCreateDepthStencilView(
                resources, GpuRenderTargetBinding{{}, 0u, depthDesc.ArraySize}) == nullptr)
        {
            return false;
        }
    }

    return true;
}

bool GpuRenderTargetSystemImpl::queueReadbackCopy(const GpuRenderTargetBinding &binding,
                                                  std::uint64_t frameIndex, std::uint64_t requestId)
{
    if (!mRenderDevice || !mGraphicsContext || !mReadbackFence ||
        !supportsDiligentRenderTargets(mBackend))
    {
        return false;
    }
    if (requestId == 0)
    {
        return false;
    }

    const auto targetIt = mRenderTargets.find(binding.target.id);
    if (targetIt == mRenderTargets.end())
    {
        return false;
    }

    const RenderTargetResources &resources  = targetIt->second;
    const GpuRenderTargetBinding normalized = normalizeBinding(binding, resources);
    if (normalized.layerCount != 1u)
    {
        return false;
    }
    if (resources.colorTexture == nullptr)
    {
        return false;
    }

    Diligent::TextureDesc stagingDesc = resources.colorTexture->GetDesc();
    const std::string stagingName     = resources.desc.debugName + ".Readback";
    stagingDesc.Name                  = stagingName.c_str();
    stagingDesc.Type                  = Diligent::RESOURCE_DIM_TEX_2D;
    stagingDesc.ArraySize             = 1u;
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
    copyAttribs.SrcSlice = normalized.firstLayer;
    mGraphicsContext->CopyTexture(copyAttribs);

    const std::uint64_t fenceValue = mNextReadbackFenceValue++;
    mGraphicsContext->EnqueueSignal(mReadbackFence, fenceValue);

    PendingReadbackCopy readbackCopy{};
    readbackCopy.requestId      = requestId;
    readbackCopy.binding        = normalized;
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
