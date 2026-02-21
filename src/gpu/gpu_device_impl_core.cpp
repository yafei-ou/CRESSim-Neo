#include "common/math_utils_runtime.h"
#include "gpu/gpu_device_impl.h"
#include "gpu/gpu_render_target_system_impl.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <iostream>
#include <limits>

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

GpuRenderTargetDesc normalizeDefaultRenderTargetForDevice(const GpuRenderTargetDesc& desc,
                                                          Diligent::ISwapChain* swapChain)
{
    GpuRenderTargetDesc normalized = desc;
    std::uint32_t fallbackWidth    = kDefaultRenderTargetWidth;
    std::uint32_t fallbackHeight   = kDefaultRenderTargetHeight;
    if (swapChain != nullptr)
    {
        const auto& swapChainDesc = swapChain->GetDesc();
        if (swapChainDesc.Width > 0)
        {
            fallbackWidth = swapChainDesc.Width;
        }
        if (swapChainDesc.Height > 0)
        {
            fallbackHeight = swapChainDesc.Height;
        }
        if (swapChainDesc.ColorBufferFormat != Diligent::TEX_FORMAT_UNKNOWN)
        {
            normalized.colorFormat = swapChainDesc.ColorBufferFormat;
        }
    }

    normalized.width =
        common::runtime_math::clampExtent(normalized.width == 0 ? fallbackWidth : normalized.width);
    normalized.height = common::runtime_math::clampExtent(
        normalized.height == 0 ? fallbackHeight : normalized.height);
    normalized.color = true;
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

} // namespace

bool GpuDeviceImpl::initialize(const GpuDeviceDesc& desc)
{
    shutdown();

    mDesc = desc;

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

    mDesc.defaultRenderTargetDesc =
        normalizeDefaultRenderTargetForDevice(mDesc.defaultRenderTargetDesc, mPrimarySwapChain);

    mRenderTargets = std::make_unique<GpuRenderTargetSystemImpl>();
    if (!mRenderTargets->initialize(mDesc.defaultRenderTargetDesc, mBackend == GpuBackend::Vulkan,
                                    mRenderDevice, mImmediateContext))
    {
        shutdown();
        return false;
    }

    mInitialized = true;
    return true;
}

void GpuDeviceImpl::shutdown()
{
    if (mRenderTargets != nullptr)
    {
        mRenderTargets->shutdown();
        mRenderTargets.reset();
    }

    if (mImmediateContext != nullptr)
    {
        mImmediateContext->SetRenderTargets(0, nullptr, nullptr,
                                            Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
        mImmediateContext->Flush();
        mImmediateContext->FinishFrame();
    }

    mImmediateContext = nullptr;
    mRenderDevice     = nullptr;
    mPrimarySwapChain = nullptr;

    mBackend     = GpuBackend::Null;
    mInitialized = false;
}

void GpuDeviceImpl::beginFrame(const common::FrameContext& frameContext)
{
    (void)frameContext;
}

GpuRenderTargetSystem& GpuDeviceImpl::renderTargetSystem()
{
    return *mRenderTargets;
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

    outContext.renderDevice     = mRenderDevice;
    outContext.immediateContext = mImmediateContext;
    if (mRenderTargets != nullptr)
    {
        mRenderTargets->fillBackendContextState(outContext);
    }
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
    swapChainDesc.DepthBufferFormat = Diligent::TEX_FORMAT_UNKNOWN;
    if (mDesc.presentation.preferredColorFormat != Diligent::TEX_FORMAT_UNKNOWN)
    {
        swapChainDesc.ColorBufferFormat = mDesc.presentation.preferredColorFormat;
    }
    else
    {
        swapChainDesc.ColorBufferFormat = Diligent::TEX_FORMAT_BGRA8_UNORM;
    }

    factoryVk->CreateSwapChainVk(mRenderDevice, mImmediateContext, swapChainDesc, window,
                                 &mPrimarySwapChain);
    if (mPrimarySwapChain == nullptr)
    {
        std::cerr << "GpuDeviceImpl: failed to create primary swapchain.\n";
        return false;
    }

    return true;
}

} // namespace cressim::neo::gpu
