#include "graphics/device/graphics_device_impl.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace cressim::neo::graphics
{

std::unique_ptr<GraphicsDevice> createGraphicsDevice()
{
    return std::make_unique<GraphicsDeviceImpl>();
}

namespace
{

std::uint32_t clampExtent(std::uint32_t value)
{
    return std::max<std::uint32_t>(value, 1u);
}

Diligent::Uint32 clampWindowId(std::uint64_t value)
{
    constexpr std::uint64_t kMax = static_cast<std::uint64_t>(std::numeric_limits<Diligent::Uint32>::max());
    return static_cast<Diligent::Uint32>(std::min<std::uint64_t>(value, kMax));
}

float clampNormalized(float value)
{
    return std::max(0.0f, std::min(value, 1.0f));
}

RenderViewport normalizeViewport(const RenderViewport& viewport)
{
    RenderViewport normalized{};
    normalized.x = clampNormalized(viewport.x);
    normalized.y = clampNormalized(viewport.y);
    normalized.width = clampNormalized(viewport.width);
    normalized.height = clampNormalized(viewport.height);

    const float maxWidth = std::max(0.0f, 1.0f - normalized.x);
    const float maxHeight = std::max(0.0f, 1.0f - normalized.y);
    normalized.width = std::min(normalized.width, maxWidth);
    normalized.height = std::min(normalized.height, maxHeight);

    if (normalized.width == 0.0f)
    {
        normalized.width = 1.0f;
        normalized.x = 0.0f;
    }
    if (normalized.height == 0.0f)
    {
        normalized.height = 1.0f;
        normalized.y = 0.0f;
    }

    return normalized;
}

} // namespace

bool GraphicsDeviceImpl::initialize(const GraphicsDeviceDesc& desc)
{
    shutdown();

    mDesc = desc;
    mDesc.initialWidth = clampExtent(mDesc.initialWidth);
    mDesc.initialHeight = clampExtent(mDesc.initialHeight);
    mShaderDirectoryResolved = false;
    mResolvedShaderDirectory.clear();
    mShaderSourceCache.clear();

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

    if (!createDebugViewerSwapChain())
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
    mCachedMeshes.clear();
    mPbrPipelineCache.clear();
    mPbrConstantBuffer = nullptr;
    mReadbackFence = nullptr;
    mNextReadbackFenceValue = 1;
    mImmediateContext = nullptr;
    mRenderDevice = nullptr;
    mSwapChain = nullptr;

    mRenderTargets.clear();
    mPendingReadbacks.clear();
    mPendingReadbackCopies.clear();
    mCompletedReadbacks.clear();
    mDefaultRenderTarget = {};
    mNextRenderTargetId = 1;

    mResolvedShaderDirectory.clear();
    mShaderSourceCache.clear();
    mShaderDirectoryResolved = false;

    mBackend = GraphicsBackend::Null;
    mInitialized = false;
}

void GraphicsDeviceImpl::resizeDefaultRenderTarget(std::uint32_t width, std::uint32_t height)
{
    mDesc.initialWidth = clampExtent(width);
    mDesc.initialHeight = clampExtent(height);

    if (!mInitialized || !isValidRenderTarget(mDefaultRenderTarget))
    {
        return;
    }

    (void)resizeRenderTarget(mDefaultRenderTarget, mDesc.initialWidth, mDesc.initialHeight);
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
    resources.viewport = normalizeViewport(RenderViewport{});

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

bool GraphicsDeviceImpl::resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height)
{
    if (!mInitialized)
    {
        return false;
    }

    const auto it = mRenderTargets.find(target.id);
    if (it == mRenderTargets.end())
    {
        return false;
    }

    RenderTargetDesc resizedDesc = it->second.desc;
    resizedDesc.width = clampExtent(width == 0 ? resizedDesc.width : width);
    resizedDesc.height = clampExtent(height == 0 ? resizedDesc.height : height);

    const bool isSwapChainDefaultTarget = (target.id == mDefaultRenderTarget.id && mSwapChain != nullptr);
    if (isSwapChainDefaultTarget)
    {
        mSwapChain->Resize(resizedDesc.width, resizedDesc.height, Diligent::SURFACE_TRANSFORM_OPTIMAL);
        const auto& swapChainDesc = mSwapChain->GetDesc();
        resizedDesc.width = clampExtent(swapChainDesc.Width);
        resizedDesc.height = clampExtent(swapChainDesc.Height);
    }

    if (mBackend == GraphicsBackend::Vulkan && !isSwapChainDefaultTarget)
    {
        RenderTargetResources resizedResources{};
        resizedResources.desc = resizedDesc;
        if (!createRenderTargetTextures(resizedDesc, resizedResources))
        {
            return false;
        }
        it->second.colorTexture = std::move(resizedResources.colorTexture);
        it->second.depthTexture = std::move(resizedResources.depthTexture);
    }

    it->second.desc = resizedDesc;

    if (target.id == mDefaultRenderTarget.id)
    {
        mDesc.initialWidth = resizedDesc.width;
        mDesc.initialHeight = resizedDesc.height;
    }

    return true;
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

    mPendingReadbacks.erase(target.id);
    mCompletedReadbacks.erase(
        std::remove_if(
            mCompletedReadbacks.begin(),
            mCompletedReadbacks.end(),
            [&](const RenderTargetReadbackEvent& event) { return event.target.id == target.id; }),
        mCompletedReadbacks.end());
    mPendingReadbackCopies.erase(
        std::remove_if(
            mPendingReadbackCopies.begin(),
            mPendingReadbackCopies.end(),
            [&](const PendingReadbackCopy& copy) { return copy.target.id == target.id; }),
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

    it->second.viewport = normalizeViewport(viewport);
}

void GraphicsDeviceImpl::beginRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext)
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
    const bool isSwapChainDefaultTarget = (target.id == mDefaultRenderTarget.id && mSwapChain != nullptr);
    if (isSwapChainDefaultTarget)
    {
        colorRtv = mSwapChain->GetCurrentBackBufferRTV();
        depthDsv = mSwapChain->GetDepthBufferDSV();
        const auto& swapChainDesc = mSwapChain->GetDesc();
        it->second.desc.width = clampExtent(swapChainDesc.Width);
        it->second.desc.height = clampExtent(swapChainDesc.Height);
        mActiveRenderTargetColorFormat = swapChainDesc.ColorBufferFormat;
    }
    else if (it->second.colorTexture != nullptr)
    {
        colorRtv = it->second.colorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET);
        mActiveRenderTargetColorFormat = it->second.colorTexture->GetDesc().Format;
    }
    else
    {
        mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    }
    if (!isSwapChainDefaultTarget && it->second.depthTexture != nullptr)
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
        constexpr float kClearColor[4] = {0.02f, 0.02f, 0.03f, 1.0f};
        mImmediateContext->ClearRenderTarget(colorRtv, kClearColor, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
    else
    {
        mImmediateContext->SetRenderTargets(0, nullptr, depthDsv, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    if (depthDsv != nullptr)
    {
        mImmediateContext->ClearDepthStencil(depthDsv, Diligent::CLEAR_DEPTH_FLAG, 1.0f, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
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

    const auto pendingReadbackIt = mPendingReadbacks.find(target.id);
    if (pendingReadbackIt == mPendingReadbacks.end())
    {
        return;
    }

    mPendingReadbacks.erase(pendingReadbackIt);

    if (mBackend == GraphicsBackend::Vulkan && mImmediateContext != nullptr)
    {
        mImmediateContext->SetRenderTargets(0, nullptr, nullptr, Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
    }

    if (queueReadbackCopy(target, frameContext.frameIndex))
    {
        return;
    }

    // Fallback path for targets/backends without pixel payload support.
    RenderTargetReadbackEvent event{};
    event.target = target;
    event.frameIndex = frameContext.frameIndex;
    mCompletedReadbacks.push_back(std::move(event));
}

GraphicsBackend GraphicsDeviceImpl::backend() const
{
    return mBackend;
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

bool GraphicsDeviceImpl::createDebugViewerSwapChain()
{
    mSwapChain = nullptr;
    if (!mDesc.debugViewer.enabled)
    {
        return true;
    }
    if (mBackend != GraphicsBackend::Vulkan || mRenderDevice == nullptr || mImmediateContext == nullptr)
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
    if (mDesc.debugViewer.nativeWindow == nullptr)
    {
        return false;
    }
    window.hWnd = mDesc.debugViewer.nativeWindow;
#elif PLATFORM_LINUX
    if (mDesc.debugViewer.nativeWindowId == 0)
    {
        return false;
    }
    window.WindowId = clampWindowId(mDesc.debugViewer.nativeWindowId);
    if (mDesc.debugViewer.nativeConnection != nullptr)
    {
        window.pXCBConnection = mDesc.debugViewer.nativeConnection;
    }
    else if (mDesc.debugViewer.nativeDisplay != nullptr)
    {
        window.pDisplay = mDesc.debugViewer.nativeDisplay;
    }
    else
    {
        return false;
    }
#elif PLATFORM_MACOS
    if (mDesc.debugViewer.nativeWindow == nullptr)
    {
        return false;
    }
    window.pNSView = mDesc.debugViewer.nativeWindow;
#else
    return false;
#endif

    Diligent::SwapChainDesc swapChainDesc{};
    swapChainDesc.Width = mDesc.initialWidth;
    swapChainDesc.Height = mDesc.initialHeight;
    factoryVk->CreateSwapChainVk(mRenderDevice, mImmediateContext, swapChainDesc, window, &mSwapChain);
    if (mSwapChain == nullptr)
    {
        return false;
    }

    const auto& activeDesc = mSwapChain->GetDesc();
    mDesc.initialWidth = clampExtent(activeDesc.Width);
    mDesc.initialHeight = clampExtent(activeDesc.Height);
    return true;
}

bool GraphicsDeviceImpl::createDefaultRenderTarget()
{
    if (mSwapChain != nullptr)
    {
        const common::ResourceId id = mNextRenderTargetId++;
        RenderTargetResources resources{};
        resources.viewport = normalizeViewport(RenderViewport{});
        resources.desc.width = mDesc.initialWidth;
        resources.desc.height = mDesc.initialHeight;
        resources.desc.color = true;
        resources.desc.depth = true;
        resources.desc.shaderReadable = false;
        resources.desc.cpuReadback = false;
        resources.desc.debugName = "CRESSimNeo.DebugViewer.Default";
        mRenderTargets.emplace(id, std::move(resources));
        mDefaultRenderTarget = RenderTargetHandle{id};
        return true;
    }

    RenderTargetDesc defaultDesc{};
    defaultDesc.width = mDesc.initialWidth;
    defaultDesc.height = mDesc.initialHeight;
    defaultDesc.color = true;
    defaultDesc.depth = true;
    defaultDesc.shaderReadable = true;
    defaultDesc.debugName = "CRESSimNeo.Headless.Default";

    mDefaultRenderTarget = createRenderTarget(defaultDesc);
    return isValidRenderTarget(mDefaultRenderTarget);
}

RenderTargetDesc GraphicsDeviceImpl::normalizeTargetDesc(const RenderTargetDesc& desc) const
{
    RenderTargetDesc normalized = desc;
    normalized.width = clampExtent(normalized.width == 0 ? mDesc.initialWidth : normalized.width);
    normalized.height = clampExtent(normalized.height == 0 ? mDesc.initialHeight : normalized.height);
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
        Diligent::TextureDesc colorDesc{};
        const std::string colorName = desc.debugName + ".Color";
        colorDesc.Name = colorName.c_str();
        colorDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        colorDesc.Width = desc.width;
        colorDesc.Height = desc.height;
        colorDesc.MipLevels = 1;
        colorDesc.ArraySize = 1;
        colorDesc.Format = Diligent::TEX_FORMAT_RGBA8_UNORM;
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
        Diligent::TextureDesc depthDesc{};
        const std::string depthName = desc.debugName + ".Depth";
        depthDesc.Name = depthName.c_str();
        depthDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        depthDesc.Width = desc.width;
        depthDesc.Height = desc.height;
        depthDesc.MipLevels = 1;
        depthDesc.ArraySize = 1;
        depthDesc.Format = Diligent::TEX_FORMAT_D32_FLOAT;
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
