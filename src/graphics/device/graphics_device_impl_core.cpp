#include "graphics/device/graphics_device_impl.h"

#include "common/math_utils_runtime.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"

#include <algorithm>
#include <utility>

namespace cressim::neo::graphics
{

std::unique_ptr<GraphicsDevice> createGraphicsDevice()
{
    return std::make_unique<GraphicsDeviceImpl>();
}

namespace
{

Diligent::TEXTURE_FORMAT toDiligentColorFormat(RenderTargetColorFormat format)
{
    switch (format)
    {
    case RenderTargetColorFormat::Rgba8Unorm:
        return Diligent::TEX_FORMAT_RGBA8_UNORM;
    case RenderTargetColorFormat::Bgra8Unorm:
        return Diligent::TEX_FORMAT_BGRA8_UNORM;
    default:
        return Diligent::TEX_FORMAT_UNKNOWN;
    }
}

Diligent::TEXTURE_FORMAT toDiligentDepthFormat(RenderTargetDepthFormat format)
{
    switch (format)
    {
    case RenderTargetDepthFormat::D32Float:
        return Diligent::TEX_FORMAT_D32_FLOAT;
    default:
        return Diligent::TEX_FORMAT_UNKNOWN;
    }
}

bool requiresTextureRecreate(const RenderTargetDesc& currentDesc, const RenderTargetDesc& updatedDesc)
{
    return currentDesc.width != updatedDesc.width ||
        currentDesc.height != updatedDesc.height ||
        currentDesc.color != updatedDesc.color ||
        currentDesc.depth != updatedDesc.depth ||
        currentDesc.colorFormat != updatedDesc.colorFormat ||
        currentDesc.depthFormat != updatedDesc.depthFormat ||
        currentDesc.shaderReadable != updatedDesc.shaderReadable;
}

} // namespace

bool GraphicsDeviceImpl::initialize(const GraphicsDeviceDesc& desc)
{
    shutdown();

    mDesc = desc;
    mDesc.defaultRenderTargetDesc = normalizeDefaultRenderTargetDesc(mDesc.defaultRenderTargetDesc);

    if (mDesc.preferredBackend == GraphicsBackend::Null)
    {
        mBackend = GraphicsBackend::Null;
        mInitialized = true;
        if (!createDefaultRenderTarget())
        {
            shutdown();
            return false;
        }
        return mInitialized;
    }

    // Vulkan-only backend path.
    if (mDesc.preferredBackend != GraphicsBackend::Vulkan)
    {
        return false;
    }

    if (!initializeVulkan())
    {
        shutdown();
        return false;
    }

    mInitialized = true;
    if (!createDefaultRenderTarget())
    {
        shutdown();
        return false;
    }

    return mInitialized;
}

void GraphicsDeviceImpl::shutdown()
{
    mHasActiveRenderTarget = false;
    mActiveRenderTargetHasDepth = false;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    mActiveRenderTarget = {};
    mReadbackFence = nullptr;
    mNextReadbackRequestId = 1;
    mNextReadbackFenceValue = 1;
    mImmediateContext = nullptr;
    mRenderDevice = nullptr;

    mRenderTargets.clear();
    mPendingReadbackRequests.clear();
    mPendingReadbackCopies.clear();
    mCompletedReadbacks.clear();
    mDefaultRenderTarget = {};
    mNextRenderTargetId = 1;

    mBackend = GraphicsBackend::Null;
    mInitialized = false;
}

RenderTargetHandle GraphicsDeviceImpl::createRenderTarget(const RenderTargetDesc& desc)
{
    if (!mInitialized)
    {
        return {};
    }

    if (mBackend == GraphicsBackend::Vulkan && !mRenderDevice)
    {
        return {};
    }

    RenderTargetResources resources{};
    resources.desc = normalizeTargetDesc(desc);
    resources.viewport = common::runtime_math::normalizeViewport(RenderViewport{});
    resources.colorFormat = resources.desc.colorFormat;
    resources.depthFormat = resources.desc.depthFormat;

    if (mBackend == GraphicsBackend::Vulkan)
    {
        if (!createRenderTargetTextures(resources.desc, resources))
        {
            return {};
        }
    }

    const common::ResourceId id = mNextRenderTargetId++;
    mRenderTargets.emplace(id, std::move(resources));
    return RenderTargetHandle{id};
}

RenderTargetUpdateResult GraphicsDeviceImpl::resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height)
{
    if (!mInitialized)
    {
        return RenderTargetUpdateResult::Failed;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return RenderTargetUpdateResult::Failed;
    }

    RenderTargetDesc resizedDesc = it->second.desc;
    resizedDesc.width = common::runtime_math::clampExtent(width == 0 ? resizedDesc.width : width);
    resizedDesc.height = common::runtime_math::clampExtent(height == 0 ? resizedDesc.height : height);
    if (resizedDesc.width == it->second.desc.width && resizedDesc.height == it->second.desc.height)
    {
        return RenderTargetUpdateResult::Unchanged;
    }

    return reconfigureRenderTarget(target, resizedDesc);
}

RenderTargetUpdateResult GraphicsDeviceImpl::reconfigureRenderTarget(RenderTargetHandle target, const RenderTargetDesc& desc)
{
    if (!mInitialized)
    {
        return RenderTargetUpdateResult::Failed;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return RenderTargetUpdateResult::Failed;
    }

    RenderTargetDesc updatedDesc = normalizeTargetDesc(desc);
    if (target.id == mDefaultRenderTarget.id)
    {
        updatedDesc = normalizeDefaultRenderTargetDesc(updatedDesc);
    }
    const bool recreateTextures = requiresTextureRecreate(it->second.desc, updatedDesc);

    RenderTargetResources updatedResources = it->second;
    updatedResources.desc = updatedDesc;
    updatedResources.colorFormat = updatedDesc.colorFormat;
    updatedResources.depthFormat = updatedDesc.depthFormat;

    if (mBackend == GraphicsBackend::Vulkan && recreateTextures)
    {
        if (mImmediateContext != nullptr)
        {
            mImmediateContext->SetRenderTargets(0, nullptr, nullptr, Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
        }

        if (!createRenderTargetTextures(updatedDesc, updatedResources))
        {
            return RenderTargetUpdateResult::Failed;
        }
    }

    it->second = std::move(updatedResources);

    if (target.id == mDefaultRenderTarget.id)
    {
        mDesc.defaultRenderTargetDesc = it->second.desc;
    }

    if (mHasActiveRenderTarget && mActiveRenderTarget.id == target.id)
    {
        mActiveRenderTargetHasDepth = it->second.desc.depth;
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        if (it->second.colorTexture != nullptr)
        {
            mActiveRenderTargetColorFormat = it->second.colorTexture->GetDesc().Format;
        }
    }

    return recreateTextures ? RenderTargetUpdateResult::Recreated : RenderTargetUpdateResult::Unchanged;
}

void GraphicsDeviceImpl::destroyRenderTarget(RenderTargetHandle target)
{
    if (target.id == common::kInvalidResourceId || target.id == mDefaultRenderTarget.id)
    {
        return;
    }

    if (mHasActiveRenderTarget && mActiveRenderTarget.id == target.id)
    {
        mHasActiveRenderTarget = false;
        mActiveRenderTargetHasDepth = false;
        mActiveRenderTarget = {};
    }

    RenderTargetColorFormat targetColorFormat = RenderTargetColorFormat::Rgba8Unorm;
    const auto targetIt = mRenderTargets.find(target.id);
    if (targetIt != mRenderTargets.end())
    {
        targetColorFormat = targetIt->second.desc.colorFormat;
    }

    auto completeRequestWithEmptyResult = [&](std::uint64_t requestId) {
        RenderTargetReadbackEvent event{};
        event.target = target;
        event.colorFormat = targetColorFormat;
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
        std::remove_if(
            mPendingReadbackCopies.begin(),
            mPendingReadbackCopies.end(),
            [&](const PendingReadbackCopy& copy) {
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

bool GraphicsDeviceImpl::isValidRenderTarget(RenderTargetHandle target) const
{
    if (target.id == common::kInvalidResourceId)
    {
        return false;
    }
    return mRenderTargets.find(target.id) != mRenderTargets.end();
}

RenderTargetHandle GraphicsDeviceImpl::defaultRenderTarget() const
{
    return mDefaultRenderTarget;
}

void GraphicsDeviceImpl::beginFrame(const common::FrameContext& frameContext)
{
    (void)frameContext;
}

void GraphicsDeviceImpl::setRenderTargetViewport(RenderTargetHandle target, const RenderViewport& viewport)
{
    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return;
    }

    it->second.viewport = common::runtime_math::normalizeViewport(viewport);
}

void GraphicsDeviceImpl::beginRenderTarget(
    RenderTargetHandle target,
    const common::FrameContext& frameContext,
    const RenderPassBeginDesc& beginDesc)
{
    (void)frameContext;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;

    if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext)
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
        colorRtv = it->second.colorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET);
        mActiveRenderTargetColorFormat = it->second.colorTexture->GetDesc().Format;
    }
    else
    {
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    }
    if (it->second.depthTexture != nullptr)
    {
        depthDsv = it->second.depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL);
    }

    if (colorRtv == nullptr && depthDsv == nullptr)
    {
        return;
    }

    if (colorRtv != nullptr)
    {
        mImmediateContext->SetRenderTargets(1, &colorRtv, depthDsv, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        if (beginDesc.clearColor)
        {
            mImmediateContext->ClearRenderTarget(colorRtv, beginDesc.clearColorValue, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        }
    }
    else
    {
        mImmediateContext->SetRenderTargets(0, nullptr, depthDsv, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    if (depthDsv != nullptr && beginDesc.clearDepth)
    {
        mImmediateContext->ClearDepthStencil(
            depthDsv,
            Diligent::CLEAR_DEPTH_FLAG,
            beginDesc.clearDepthValue,
            0,
            Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    const float targetWidth = static_cast<float>(it->second.desc.width);
    const float targetHeight = static_cast<float>(it->second.desc.height);
    const RenderViewport viewport = it->second.viewport;

    Diligent::Viewport diligentViewport{};
    diligentViewport.TopLeftX = viewport.x * targetWidth;
    diligentViewport.TopLeftY = viewport.y * targetHeight;
    diligentViewport.Width = viewport.width * targetWidth;
    diligentViewport.Height = viewport.height * targetHeight;
    diligentViewport.MinDepth = 0.0f;
    diligentViewport.MaxDepth = 1.0f;
    mImmediateContext->SetViewports(1, &diligentViewport, it->second.desc.width, it->second.desc.height);

    mActiveRenderTarget = target;
    mHasActiveRenderTarget = true;
    mActiveRenderTargetHasDepth = (depthDsv != nullptr);
}

void GraphicsDeviceImpl::endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext)
{
    if (mHasActiveRenderTarget && mActiveRenderTarget.id == target.id)
    {
        mHasActiveRenderTarget = false;
        mActiveRenderTargetHasDepth = false;
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        mActiveRenderTarget = {};
    }

    if (mBackend == GraphicsBackend::Vulkan && mImmediateContext != nullptr)
    {
        mImmediateContext->SetRenderTargets(0, nullptr, nullptr, Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
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

    // Fallback path for targets/backends without pixel payload support.
    RenderTargetReadbackEvent event{};
    event.target = target;
    event.frameIndex = frameContext.frameIndex;
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

GraphicsBackend GraphicsDeviceImpl::backend() const
{
    return mBackend;
}

bool GraphicsDeviceImpl::tryGetVulkanContext(VulkanBackendContext& outContext)
{
    outContext = {};

    if (!mInitialized || mBackend != GraphicsBackend::Vulkan || mRenderDevice == nullptr || mImmediateContext == nullptr)
    {
        return false;
    }

    outContext.renderDevice = mRenderDevice;
    outContext.immediateContext = mImmediateContext;
    outContext.hasActiveRenderTarget = mHasActiveRenderTarget;
    outContext.activeRenderTargetId = mHasActiveRenderTarget ? mActiveRenderTarget.id : common::kInvalidResourceId;
    outContext.activeRenderTargetHasDepth = mActiveRenderTargetHasDepth;
    outContext.activeRenderTargetColorFormat = mActiveRenderTargetColorFormat;
    return true;
}

bool GraphicsDeviceImpl::tryGetRenderTargetDesc(RenderTargetHandle target, RenderTargetDesc& outDesc) const
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

bool GraphicsDeviceImpl::tryGetRenderTargetColorTexture(RenderTargetHandle target, Diligent::ITexture*& outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || mBackend != GraphicsBackend::Vulkan)
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

bool GraphicsDeviceImpl::tryGetRenderTargetDepthTexture(RenderTargetHandle target, Diligent::ITexture*& outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || mBackend != GraphicsBackend::Vulkan)
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

const std::string& GraphicsDeviceImpl::shaderSourceDirectory() const
{
    return mDesc.shaderDirectory;
}

bool GraphicsDeviceImpl::initializeVulkan()
{
    Diligent::IEngineFactoryVk* factoryVk = Diligent::LoadAndGetEngineFactoryVk();
    if (factoryVk == nullptr)
    {
        return false;
    }

    Diligent::EngineVkCreateInfo engineCreateInfo{};
    engineCreateInfo.EnableValidation = static_cast<Diligent::Bool>(mDesc.enableValidation ? 1 : 0);
    factoryVk->CreateDeviceAndContextsVk(engineCreateInfo, &mRenderDevice, &mImmediateContext);
    if (!mRenderDevice || !mImmediateContext)
    {
        return false;
    }

    Diligent::FenceDesc readbackFenceDesc{};
    readbackFenceDesc.Name = "CRESSimNeo.ReadbackFence";
    readbackFenceDesc.Type = Diligent::FENCE_TYPE_CPU_WAIT_ONLY;
    mRenderDevice->CreateFence(readbackFenceDesc, &mReadbackFence);
    if (mReadbackFence == nullptr)
    {
        return false;
    }

    mBackend = GraphicsBackend::Vulkan;
    return true;
}

bool GraphicsDeviceImpl::createDefaultRenderTarget()
{
    RenderTargetDesc defaultDesc = normalizeDefaultRenderTargetDesc(mDesc.defaultRenderTargetDesc);
    mDesc.defaultRenderTargetDesc = defaultDesc;

    mDefaultRenderTarget = createRenderTarget(defaultDesc);
    return isValidRenderTarget(mDefaultRenderTarget);
}


constexpr std::uint32_t kDefaultRenderTargetWidth = 1280u;
constexpr std::uint32_t kDefaultRenderTargetHeight = 720u;

RenderTargetDesc GraphicsDeviceImpl::normalizeDefaultRenderTargetDesc(const RenderTargetDesc& desc) const
{
    RenderTargetDesc normalized = desc;
    normalized.width = common::runtime_math::clampExtent(normalized.width == 0 ? kDefaultRenderTargetWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(normalized.height == 0 ? kDefaultRenderTargetHeight : normalized.height);
    normalized.color = true;
    if (normalized.debugName.empty())
    {
        normalized.debugName = "CRESSimNeo.Default";
    }
    return normalized;
}

RenderTargetDesc GraphicsDeviceImpl::normalizeTargetDesc(const RenderTargetDesc& desc) const
{
    RenderTargetDesc normalized = desc;
    const std::uint32_t fallbackWidth =
        common::runtime_math::clampExtent(mDesc.defaultRenderTargetDesc.width == 0 ? kDefaultRenderTargetWidth : mDesc.defaultRenderTargetDesc.width);
    const std::uint32_t fallbackHeight =
        common::runtime_math::clampExtent(mDesc.defaultRenderTargetDesc.height == 0 ? kDefaultRenderTargetHeight : mDesc.defaultRenderTargetDesc.height);
    normalized.width = common::runtime_math::clampExtent(normalized.width == 0 ? fallbackWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(normalized.height == 0 ? fallbackHeight : normalized.height);
    if (!normalized.color && !normalized.depth)
    {
        normalized.color = true;
    }
    if (normalized.debugName.empty())
    {
        normalized.debugName = "CRESSimNeo.RenderTarget";
    }
    return normalized;
}

bool GraphicsDeviceImpl::createRenderTargetTextures(const RenderTargetDesc& desc, RenderTargetResources& resources)
{
    if (!mRenderDevice || (!desc.color && !desc.depth))
    {
        return false;
    }

    resources.colorTexture = nullptr;
    resources.depthTexture = nullptr;

    if (desc.color)
    {
        const Diligent::TEXTURE_FORMAT colorFormat = toDiligentColorFormat(resources.colorFormat);
        if (colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
        {
            return false;
        }

        Diligent::TextureDesc colorDesc{};
        const std::string colorName = desc.debugName + ".Color";
        colorDesc.Name = colorName.c_str();
        colorDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        colorDesc.Width = desc.width;
        colorDesc.Height = desc.height;
        colorDesc.MipLevels = 1;
        colorDesc.ArraySize = 1;
        colorDesc.Format = colorFormat;
        colorDesc.BindFlags = Diligent::BIND_RENDER_TARGET;
        if (desc.shaderReadable)
        {
            colorDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
        }
        colorDesc.Usage = Diligent::USAGE_DEFAULT;

        mRenderDevice->CreateTexture(colorDesc, nullptr, &resources.colorTexture);
        if (!resources.colorTexture || resources.colorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET) == nullptr)
        {
            return false;
        }
    }

    if (desc.depth)
    {
        const Diligent::TEXTURE_FORMAT depthFormat = toDiligentDepthFormat(resources.depthFormat);
        if (depthFormat == Diligent::TEX_FORMAT_UNKNOWN)
        {
            return false;
        }

        Diligent::TextureDesc depthDesc{};
        const std::string depthName = desc.debugName + ".Depth";
        depthDesc.Name = depthName.c_str();
        depthDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        depthDesc.Width = desc.width;
        depthDesc.Height = desc.height;
        depthDesc.MipLevels = 1;
        depthDesc.ArraySize = 1;
        depthDesc.Format = depthFormat;
        depthDesc.BindFlags = Diligent::BIND_DEPTH_STENCIL;
        if (desc.shaderReadable)
        {
            depthDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
        }
        depthDesc.Usage = Diligent::USAGE_DEFAULT;

        mRenderDevice->CreateTexture(depthDesc, nullptr, &resources.depthTexture);
        if (!resources.depthTexture || resources.depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL) == nullptr)
        {
            return false;
        }
    }

    return true;
}

} // namespace cressim::neo::graphics
