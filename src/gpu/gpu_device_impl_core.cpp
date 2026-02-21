#include "common/math_utils_runtime.h"
#include "gpu/gpu_device_impl.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>

namespace cressim::neo::gpu
{

std::unique_ptr<GpuDevice> createGpuDevice()
{
    return std::make_unique<GpuDeviceImpl>();
}

namespace
{

constexpr std::uint32_t kDefaultRenderTargetWidth  = 1280u;
constexpr std::uint32_t kDefaultRenderTargetHeight = 720u;

Diligent::Uint32 clampWindowId(std::uint64_t value)
{
    constexpr std::uint64_t kMax =
        static_cast<std::uint64_t>(std::numeric_limits<Diligent::Uint32>::max());
    return static_cast<Diligent::Uint32>(std::min<std::uint64_t>(value, kMax));
}

bool requiresTextureRecreate(const GpuRenderTargetDesc& currentDesc,
                             const GpuRenderTargetDesc& updatedDesc)
{
    return currentDesc.width != updatedDesc.width || currentDesc.height != updatedDesc.height ||
           currentDesc.color != updatedDesc.color || currentDesc.depth != updatedDesc.depth ||
           currentDesc.colorFormat != updatedDesc.colorFormat ||
           currentDesc.depthFormat != updatedDesc.depthFormat ||
           currentDesc.shaderReadable != updatedDesc.shaderReadable;
}

} // namespace

bool GpuDeviceImpl::initialize(const GpuDeviceDesc& desc)
{
    shutdown();

    mDesc = desc;

    // Vulkan-only backend path.
    if (mDesc.preferredBackend != GpuBackend::Vulkan)
    {
        return false;
    }

    if (!initializeVulkan())
    {
        shutdown();
        return false;
    }

    if (mDesc.presentation.enabled && !createPrimarySwapChain())
    {
        shutdown();
        return false;
    }

    mDesc.defaultRenderTargetDesc = normalizeDefaultRenderTargetDesc(mDesc.defaultRenderTargetDesc);

    mInitialized = true;
    if (!createDefaultRenderTarget())
    {
        shutdown();
        return false;
    }

    return mInitialized;
}

void GpuDeviceImpl::shutdown()
{
    if (mImmediateContext != nullptr)
    {
        mImmediateContext->SetRenderTargets(0, nullptr, nullptr,
                                            Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
        mImmediateContext->Flush();
        mImmediateContext->FinishFrame();
    }

    mHasActiveRenderTarget         = false;
    mActiveRenderTargetHasDepth    = false;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    mActiveRenderTarget            = {};
    mReadbackFence                 = nullptr;
    mNextReadbackRequestId         = 1;
    mNextReadbackFenceValue        = 1;
    mImmediateContext              = nullptr;
    mRenderDevice                  = nullptr;
    mPrimarySwapChain              = nullptr;

    mRenderTargets.clear();
    mPendingReadbackRequests.clear();
    mPendingReadbackCopies.clear();
    mCompletedReadbacks.clear();
    mDefaultRenderTarget = {};
    mNextRenderTargetId  = 1;

    mBackend     = GpuBackend::Null;
    mInitialized = false;
}

GpuRenderTargetHandle GpuDeviceImpl::createRenderTarget(const GpuRenderTargetDesc& desc)
{
    if (!mInitialized)
    {
        return {};
    }

    if (mBackend == GpuBackend::Vulkan && !mRenderDevice)
    {
        return {};
    }

    RenderTargetResources resources{};
    resources.desc        = normalizeTargetDesc(desc);
    resources.viewport    = common::runtime_math::normalizeViewport(GpuRenderViewport{});
    resources.colorFormat = resources.desc.colorFormat;
    resources.depthFormat = resources.desc.depthFormat;

    if (mBackend == GpuBackend::Vulkan)
    {
        if (!createRenderTargetTextures(resources.desc, resources))
        {
            return {};
        }
    }

    const common::ResourceId id = mNextRenderTargetId++;
    mRenderTargets.emplace(id, std::move(resources));
    return GpuRenderTargetHandle{id};
}

GpuRenderTargetUpdateResult GpuDeviceImpl::resizeRenderTarget(GpuRenderTargetHandle target,
                                                              std::uint32_t width,
                                                              std::uint32_t height)
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

GpuRenderTargetUpdateResult GpuDeviceImpl::reconfigureRenderTarget(GpuRenderTargetHandle target,
                                                                   const GpuRenderTargetDesc& desc)
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

    if (mBackend == GpuBackend::Vulkan && recreateTextures)
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
        mDesc.defaultRenderTargetDesc = it->second.desc;
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

void GpuDeviceImpl::destroyRenderTarget(GpuRenderTargetHandle target)
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

bool GpuDeviceImpl::isValidRenderTarget(GpuRenderTargetHandle target) const
{
    if (target.id == common::kInvalidResourceId)
    {
        return false;
    }
    return mRenderTargets.find(target.id) != mRenderTargets.end();
}

GpuRenderTargetHandle GpuDeviceImpl::defaultRenderTarget() const
{
    return mDefaultRenderTarget;
}

void GpuDeviceImpl::beginFrame(const common::FrameContext& frameContext)
{
    (void)frameContext;
}

void GpuDeviceImpl::setRenderTargetViewport(GpuRenderTargetHandle target,
                                            const GpuRenderViewport& viewport)
{
    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return;
    }

    it->second.viewport = common::runtime_math::normalizeViewport(viewport);
}

void GpuDeviceImpl::beginRenderTarget(GpuRenderTargetHandle target,
                                      const common::FrameContext& frameContext,
                                      const GpuRenderPassBeginDesc& beginDesc)
{
    (void)frameContext;
    mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;

    if (!mInitialized || mBackend != GpuBackend::Vulkan || !mImmediateContext)
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

void GpuDeviceImpl::endRenderTarget(GpuRenderTargetHandle target,
                                    const common::FrameContext& frameContext)
{
    if (mHasActiveRenderTarget && mActiveRenderTarget.id == target.id)
    {
        mHasActiveRenderTarget         = false;
        mActiveRenderTargetHasDepth    = false;
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        mActiveRenderTarget            = {};
    }

    if (mBackend == GpuBackend::Vulkan && mImmediateContext != nullptr)
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

    // Fallback path for targets/backends without pixel payload support.
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

GpuBackend GpuDeviceImpl::backend() const
{
    return mBackend;
}

bool GpuDeviceImpl::tryGetBackendContext(GpuBackendContext& outContext)
{
    outContext = GpuBackendContext{};

    if (!mInitialized || mBackend != GpuBackend::Vulkan || mRenderDevice == nullptr ||
        mImmediateContext == nullptr)
    {
        return false;
    }

    outContext.renderDevice          = mRenderDevice;
    outContext.immediateContext      = mImmediateContext;
    outContext.hasActiveRenderTarget = mHasActiveRenderTarget;
    outContext.activeRenderTargetId =
        mHasActiveRenderTarget ? mActiveRenderTarget.id : common::kInvalidResourceId;
    outContext.activeRenderTargetHasDepth    = mActiveRenderTargetHasDepth;
    outContext.activeRenderTargetColorFormat = mActiveRenderTargetColorFormat;
    return true;
}

bool GpuDeviceImpl::tryGetRenderTargetDesc(GpuRenderTargetHandle target,
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

bool GpuDeviceImpl::tryGetRenderTargetColorTexture(GpuRenderTargetHandle target,
                                                   Diligent::ITexture*& outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || mBackend != GpuBackend::Vulkan)
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

bool GpuDeviceImpl::tryGetRenderTargetDepthTexture(GpuRenderTargetHandle target,
                                                   Diligent::ITexture*& outTexture)
{
    outTexture = nullptr;

    if (!mInitialized || mBackend != GpuBackend::Vulkan)
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

const std::string& GpuDeviceImpl::shaderSourceDirectory() const
{
    return mDesc.shaderDirectory;
}

bool GpuDeviceImpl::initializeVulkan()
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

    mBackend = GpuBackend::Vulkan;
    return true;
}

bool GpuDeviceImpl::createPrimarySwapChain()
{
    if (!mDesc.presentation.enabled)
    {
        return true;
    }
    if (mBackend != GpuBackend::Vulkan || mRenderDevice == nullptr || mImmediateContext == nullptr)
    {
        return false;
    }

    Diligent::IEngineFactoryVk* factoryVk = Diligent::LoadAndGetEngineFactoryVk();
    if (factoryVk == nullptr)
    {
        return false;
    }

    Diligent::NativeWindow window{};
#if PLATFORM_WIN32
    if (mDesc.presentation.nativeWindow == nullptr)
    {
        std::cerr << "GpuDeviceImpl: presentation nativeWindow must be set on Win32.\n";
        return false;
    }
    window.hWnd = mDesc.presentation.nativeWindow;
#elif PLATFORM_LINUX
    if (mDesc.presentation.nativeWindowId == 0)
    {
        std::cerr << "GpuDeviceImpl: presentation nativeWindowId must be set on Linux.\n";
        return false;
    }
    window.WindowId = clampWindowId(mDesc.presentation.nativeWindowId);
    if (mDesc.presentation.nativeConnection != nullptr)
    {
        window.pXCBConnection = mDesc.presentation.nativeConnection;
    }
    else if (mDesc.presentation.nativeDisplay != nullptr)
    {
        window.pDisplay = mDesc.presentation.nativeDisplay;
    }
    else
    {
        std::cerr << "GpuDeviceImpl: presentation native display/connection is missing on Linux.\n";
        return false;
    }
#elif PLATFORM_MACOS
    if (mDesc.presentation.nativeWindow == nullptr)
    {
        std::cerr << "GpuDeviceImpl: presentation nativeWindow must be set on macOS.\n";
        return false;
    }
    window.pNSView = mDesc.presentation.nativeWindow;
#else
    std::cerr << "GpuDeviceImpl: presentation is unsupported on this platform.\n";
    return false;
#endif

    Diligent::SwapChainDesc swapChainDesc{};
    swapChainDesc.Width = common::runtime_math::clampExtent(
        mDesc.defaultRenderTargetDesc.width == 0 ? kDefaultRenderTargetWidth
                                                 : mDesc.defaultRenderTargetDesc.width);
    swapChainDesc.Height = common::runtime_math::clampExtent(
        mDesc.defaultRenderTargetDesc.height == 0 ? kDefaultRenderTargetHeight
                                                  : mDesc.defaultRenderTargetDesc.height);
    swapChainDesc.DepthBufferFormat                     = Diligent::TEX_FORMAT_UNKNOWN;
    const Diligent::TEXTURE_FORMAT preferredColorFormat = mDesc.presentation.preferredColorFormat;
    if (preferredColorFormat != Diligent::TEX_FORMAT_UNKNOWN)
    {
        swapChainDesc.ColorBufferFormat = preferredColorFormat;
    }
    else
    {
        // TODO: always use non-sRGB color format
        // This is needed because our shaders now use manual gamma
        // Flags or shader variants should be implemented
        swapChainDesc.ColorBufferFormat = Diligent::TEX_FORMAT_BGRA8_UNORM;
    }

    factoryVk->CreateSwapChainVk(mRenderDevice, mImmediateContext, swapChainDesc, window,
                                 &mPrimarySwapChain);
    if (mPrimarySwapChain == nullptr)
    {
        std::cerr << "GpuDeviceImpl: failed to create primary swapchain.\n";
        return false;
    }

    const Diligent::TEXTURE_FORMAT actualSwapChainFormat =
        mPrimarySwapChain->GetDesc().ColorBufferFormat;

    // Force the default offscreen color target to the primary swapchain format.
    mDesc.defaultRenderTargetDesc.colorFormat = actualSwapChainFormat;

    return true;
}

bool GpuDeviceImpl::createDefaultRenderTarget()
{
    GpuRenderTargetDesc defaultDesc =
        normalizeDefaultRenderTargetDesc(mDesc.defaultRenderTargetDesc);
    mDesc.defaultRenderTargetDesc = defaultDesc;

    mDefaultRenderTarget = createRenderTarget(defaultDesc);
    return isValidRenderTarget(mDefaultRenderTarget);
}

GpuRenderTargetDesc GpuDeviceImpl::normalizeDefaultRenderTargetDesc(
    const GpuRenderTargetDesc& desc) const
{
    GpuRenderTargetDesc normalized = desc;
    std::uint32_t fallbackWidth    = kDefaultRenderTargetWidth;
    std::uint32_t fallbackHeight   = kDefaultRenderTargetHeight;
    if (mPrimarySwapChain != nullptr)
    {
        const auto& swapChainDesc = mPrimarySwapChain->GetDesc();
        if (swapChainDesc.Width > 0)
        {
            fallbackWidth = swapChainDesc.Width;
        }
        if (swapChainDesc.Height > 0)
        {
            fallbackHeight = swapChainDesc.Height;
        }
    }
    normalized.width =
        common::runtime_math::clampExtent(normalized.width == 0 ? fallbackWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(
        normalized.height == 0 ? fallbackHeight : normalized.height);
    normalized.color = true;
    if (mPrimarySwapChain != nullptr)
    {
        const Diligent::TEXTURE_FORMAT swapChainFormat =
            mPrimarySwapChain->GetDesc().ColorBufferFormat;
        if (swapChainFormat == Diligent::TEX_FORMAT_UNKNOWN)
        {
            normalized.colorFormat = Diligent::TEX_FORMAT_RGBA8_UNORM;
        }
        else
        {
            normalized.colorFormat = swapChainFormat;
        }
    }
    else if (normalized.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        normalized.colorFormat = Diligent::TEX_FORMAT_RGBA8_UNORM;
    }
    else if (normalized.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        normalized.colorFormat = Diligent::TEX_FORMAT_RGBA8_UNORM;
    }
    if (normalized.debugName.empty())
    {
        normalized.debugName = "CRESSimNeo.Default";
    }
    return normalized;
}

GpuRenderTargetDesc GpuDeviceImpl::normalizeTargetDesc(const GpuRenderTargetDesc& desc) const
{
    GpuRenderTargetDesc normalized    = desc;
    const std::uint32_t fallbackWidth = common::runtime_math::clampExtent(
        mDesc.defaultRenderTargetDesc.width == 0 ? kDefaultRenderTargetWidth
                                                 : mDesc.defaultRenderTargetDesc.width);
    const std::uint32_t fallbackHeight = common::runtime_math::clampExtent(
        mDesc.defaultRenderTargetDesc.height == 0 ? kDefaultRenderTargetHeight
                                                  : mDesc.defaultRenderTargetDesc.height);
    normalized.width =
        common::runtime_math::clampExtent(normalized.width == 0 ? fallbackWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(
        normalized.height == 0 ? fallbackHeight : normalized.height);
    if (normalized.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        normalized.colorFormat = mDesc.defaultRenderTargetDesc.colorFormat;
    }
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

bool GpuDeviceImpl::createRenderTargetTextures(const GpuRenderTargetDesc& desc,
                                               RenderTargetResources& resources)
{
    if (!mRenderDevice || (!desc.color && !desc.depth))
    {
        return false;
    }

    resources.colorTexture = nullptr;
    resources.depthTexture = nullptr;

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
        colorDesc.Type              = Diligent::RESOURCE_DIM_TEX_2D;
        colorDesc.Width             = desc.width;
        colorDesc.Height            = desc.height;
        colorDesc.MipLevels         = 1;
        colorDesc.ArraySize         = 1;
        colorDesc.Format            = colorFormat;
        colorDesc.BindFlags         = Diligent::BIND_RENDER_TARGET;
        if (desc.shaderReadable)
        {
            colorDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
        }
        colorDesc.Usage = Diligent::USAGE_DEFAULT;

        mRenderDevice->CreateTexture(colorDesc, nullptr, &resources.colorTexture);
        if (!resources.colorTexture ||
            resources.colorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET) == nullptr)
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
        depthDesc.Type              = Diligent::RESOURCE_DIM_TEX_2D;
        depthDesc.Width             = desc.width;
        depthDesc.Height            = desc.height;
        depthDesc.MipLevels         = 1;
        depthDesc.ArraySize         = 1;
        depthDesc.Format            = depthFormat;
        depthDesc.BindFlags         = Diligent::BIND_DEPTH_STENCIL;
        if (desc.shaderReadable)
        {
            depthDesc.BindFlags |= Diligent::BIND_SHADER_RESOURCE;
        }
        depthDesc.Usage = Diligent::USAGE_DEFAULT;

        mRenderDevice->CreateTexture(depthDesc, nullptr, &resources.depthTexture);
        if (!resources.depthTexture ||
            resources.depthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL) == nullptr)
        {
            return false;
        }
    }

    return true;
}

} // namespace cressim::neo::gpu
