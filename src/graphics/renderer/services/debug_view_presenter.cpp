#include "graphics/renderer/services/debug_view_presenter.h"

#include "common/math_utils_runtime.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <limits>

namespace cressim::neo::graphics::detail
{

namespace
{
Diligent::Uint32 clampWindowId(std::uint64_t value)
{
    constexpr std::uint64_t kMax = static_cast<std::uint64_t>(std::numeric_limits<Diligent::Uint32>::max());
    return static_cast<Diligent::Uint32>(std::min<std::uint64_t>(value, kMax));
}

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

bool tryFromDiligentColorFormat(Diligent::TEXTURE_FORMAT format, RenderTargetColorFormat& outFormat)
{
    switch (format)
    {
    case Diligent::TEX_FORMAT_RGBA8_UNORM:
        outFormat = RenderTargetColorFormat::Rgba8Unorm;
        return true;
    case Diligent::TEX_FORMAT_BGRA8_UNORM:
        outFormat = RenderTargetColorFormat::Bgra8Unorm;
        return true;
    default:
        outFormat = RenderTargetColorFormat::Rgba8Unorm;
        return false;
    }
}

} // namespace

DebugViewPresenter::DebugViewPresenter(GraphicsDeviceImpl& device, const RendererDesc::DebugViewerDesc& desc) :
    mDevice(device),
    mDesc(desc)
{
}

bool DebugViewPresenter::initialize()
{
    if (!mDesc.enabled)
    {
        return true;
    }
    if (mDevice.backend() != GraphicsBackend::Vulkan)
    {
        return true;
    }

    RenderTargetDesc defaultDesc{};
    if (!mDevice.tryGetRenderTargetDesc(mDevice.defaultRenderTarget(), defaultDesc))
    {
        return false;
    }
    if (!defaultDesc.color)
    {
        return false;
    }

    const Diligent::TEXTURE_FORMAT requestedColorFormat = toDiligentColorFormat(defaultDesc.colorFormat);
    if (requestedColorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return false;
    }

    if (!createSwapChain(defaultDesc.width, defaultDesc.height, requestedColorFormat))
    {
        return false;
    }

    return true;
}

bool DebugViewPresenter::present(RenderTargetHandle sourceTarget)
{
    if (!mDesc.enabled || mSwapChain == nullptr)
    {
        return false;
    }

    GraphicsDeviceImpl::VulkanBackendContext backendContext{};
    if (!mDevice.tryGetVulkanContext(backendContext) || backendContext.immediateContext == nullptr)
    {
        return false;
    }

    Diligent::ITexture* sourceTexture = nullptr;
    if (!mDevice.tryGetRenderTargetColorTexture(sourceTarget, sourceTexture) || sourceTexture == nullptr)
    {
        return false;
    }

    RenderTargetDesc sourceDesc{};
    if (!mDevice.tryGetRenderTargetDesc(sourceTarget, sourceDesc))
    {
        return false;
    }

    const auto& swapChainDesc = mSwapChain->GetDesc();
    if (swapChainDesc.Width != sourceDesc.width || swapChainDesc.Height != sourceDesc.height)
    {
        mSwapChain->Resize(
            common::runtime_math::clampExtent(sourceDesc.width),
            common::runtime_math::clampExtent(sourceDesc.height),
            Diligent::SURFACE_TRANSFORM_OPTIMAL);
    }

    Diligent::ITextureView* backBufferRtv = mSwapChain->GetCurrentBackBufferRTV();
    if (backBufferRtv == nullptr || backBufferRtv->GetTexture() == nullptr)
    {
        return false;
    }

    const Diligent::TEXTURE_FORMAT backBufferFormat = backBufferRtv->GetTexture()->GetDesc().Format;
    if (sourceTexture->GetDesc().Format != backBufferFormat)
    {
        RenderTargetColorFormat backBufferColorFormat = RenderTargetColorFormat::Rgba8Unorm;
        if (!tryFromDiligentColorFormat(backBufferFormat, backBufferColorFormat))
        {
            return false;
        }
        sourceDesc.colorFormat = backBufferColorFormat;
        const RenderTargetUpdateResult updateResult = mDevice.reconfigureRenderTarget(sourceTarget, sourceDesc);
        if (updateResult == RenderTargetUpdateResult::Failed)
        {
            return false;
        }
        if (!mDevice.tryGetRenderTargetColorTexture(sourceTarget, sourceTexture) || sourceTexture == nullptr)
        {
            return false;
        }
    }

    Diligent::CopyTextureAttribs copyAttribs{
        sourceTexture,
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
        backBufferRtv->GetTexture(),
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION};
    backendContext.immediateContext->CopyTexture(copyAttribs);

    mSwapChain->Present(mDesc.syncInterval);
    if (!mSwapChain->GetDesc().IsPrimary)
    {
        backendContext.immediateContext->Flush();
        backendContext.immediateContext->FinishFrame();
    }

    return true;
}

bool DebugViewPresenter::createSwapChain(std::uint32_t width, std::uint32_t height, Diligent::TEXTURE_FORMAT colorFormat)
{
    GraphicsDeviceImpl::VulkanBackendContext backendContext{};
    if (!mDevice.tryGetVulkanContext(backendContext) || backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr)
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
    if (mDesc.nativeWindow == nullptr)
    {
        return false;
    }
    window.hWnd = mDesc.nativeWindow;
#elif PLATFORM_LINUX
    if (mDesc.nativeWindowId == 0)
    {
        return false;
    }
    window.WindowId = clampWindowId(mDesc.nativeWindowId);
    if (mDesc.nativeConnection != nullptr)
    {
        window.pXCBConnection = mDesc.nativeConnection;
    }
    else if (mDesc.nativeDisplay != nullptr)
    {
        window.pDisplay = mDesc.nativeDisplay;
    }
    else
    {
        return false;
    }
#elif PLATFORM_MACOS
    if (mDesc.nativeWindow == nullptr)
    {
        return false;
    }
    window.pNSView = mDesc.nativeWindow;
#else
    return false;
#endif

    Diligent::SwapChainDesc swapChainDesc{};
    swapChainDesc.Width = common::runtime_math::clampExtent(width);
    swapChainDesc.Height = common::runtime_math::clampExtent(height);
    swapChainDesc.ColorBufferFormat = colorFormat;
    swapChainDesc.DepthBufferFormat = Diligent::TEX_FORMAT_UNKNOWN;
    factoryVk->CreateSwapChainVk(backendContext.renderDevice, backendContext.immediateContext, swapChainDesc, window, &mSwapChain);
    return mSwapChain != nullptr;
}

} // namespace cressim::neo::graphics::detail
