#include "common/logger.h"
#include "gpu/gpu_device_impl.h"
#include "gpu/gpu_render_target_system_impl.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <array>
#include <limits>

namespace cressim::neo::gpu
{

std::unique_ptr<GpuDevice> createGpuDevice()
{
    return std::make_unique<GpuDeviceImpl>();
}

namespace
{

Diligent::Uint32 clampWindowId(std::uint64_t value)
{
    constexpr std::uint64_t kMax =
        static_cast<std::uint64_t>(std::numeric_limits<Diligent::Uint32>::max());
    return static_cast<Diligent::Uint32>(std::min<std::uint64_t>(value, kMax));
}

GpuRenderTargetDesc effectiveDefaultRenderTargetDesc(const GpuDeviceDesc &deviceDesc)
{
    return normalizeDefaultRenderTargetDesc(deviceDesc.defaultRenderTargetDesc);
}

} // namespace

bool GpuDeviceImpl::initialize(const GpuDeviceDesc &desc)
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

    mRenderTargets = std::make_unique<GpuRenderTargetSystemImpl>();
    if (!mRenderTargets->initialize(mBackend == GpuBackend::Vulkan, mRenderDevice,
                                    mImmediateContext))
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
    if (mPhysicsContext != nullptr && mPhysicsContext != mImmediateContext)
    {
        mPhysicsContext->Flush();
        mPhysicsContext->FinishFrame();
    }

    mImmediateContext                   = nullptr;
    mPhysicsContext                     = nullptr;
    mRenderDevice                       = nullptr;
    mPrimarySwapChain                   = nullptr;
    mPresentationReadbackFence          = nullptr;
    mGraphicsContextId                  = 0;
    mPhysicsContextId                   = 0;
    mGraphicsQueueType                  = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mPhysicsQueueType                   = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mNextPresentationReadbackRequestId  = 1;
    mNextPresentationReadbackFenceValue = 1;
    mPendingPresentationReadbackRequests.clear();
    mPendingPresentationReadbackCopies.clear();
    mCompletedPresentationReadbacks.clear();

    mBackend     = GpuBackend::Null;
    mInitialized = false;
}

void GpuDeviceImpl::beginFrame(const common::FrameContext &frameContext)
{
    (void)frameContext;
}

GpuRenderTargetSystem &GpuDeviceImpl::renderTargetSystem()
{
    return *mRenderTargets;
}

GpuBackend GpuDeviceImpl::backend() const
{
    return mBackend;
}

bool GpuDeviceImpl::tryGetGraphicsBackendContext(GpuBackendContext &outContext)
{
    outContext = GpuBackendContext{};

    if (!mInitialized || mBackend != GpuBackend::Vulkan || mRenderDevice == nullptr ||
        mImmediateContext == nullptr)
    {
        return false;
    }

    outContext.renderDevice     = mRenderDevice;
    outContext.immediateContext = mImmediateContext;
    outContext.primarySwapChain = mPrimarySwapChain;
    if (mRenderTargets != nullptr)
    {
        mRenderTargets->fillBackendContextState(outContext);
    }
    return true;
}

bool GpuDeviceImpl::tryGetPhysicsBackendContext(GpuComputeBackendContext &outContext)
{
    outContext = GpuComputeBackendContext{};

    if (!mInitialized || mBackend != GpuBackend::Vulkan || mRenderDevice == nullptr ||
        mPhysicsContext == nullptr)
    {
        return false;
    }

    outContext.renderDevice   = mRenderDevice;
    outContext.computeContext = mPhysicsContext;
    outContext.contextId      = mPhysicsContextId;
    outContext.queueType      = mPhysicsQueueType;
    outContext.role           = GpuContextRole::Physics;
    return true;
}

bool GpuDeviceImpl::tryGetDefaultRenderTargetDesc(GpuRenderTargetDesc &outDesc) const
{
    outDesc = {};
    if (!mInitialized)
    {
        return false;
    }

    outDesc = effectiveDefaultRenderTargetDesc(mDesc);
    return true;
}

bool GpuDeviceImpl::tryGetPresentationTargetDesc(GpuPresentationTargetDesc &outDesc)
{
    outDesc = {};
    if (!mInitialized || mBackend != GpuBackend::Vulkan || mPrimarySwapChain == nullptr)
    {
        return false;
    }

    const auto &swapChainDesc = mPrimarySwapChain->GetDesc();
    outDesc.width             = swapChainDesc.Width;
    outDesc.height            = swapChainDesc.Height;
    outDesc.colorFormat       = swapChainDesc.ColorBufferFormat;
    outDesc.depthFormat       = swapChainDesc.DepthBufferFormat;
    outDesc.hasDepth          = swapChainDesc.DepthBufferFormat != Diligent::TEX_FORMAT_UNKNOWN;
    return true;
}

GpuPresentationReadbackRequest GpuDeviceImpl::requestPresentationReadback()
{
    if (!mInitialized || mBackend != GpuBackend::Vulkan || mPrimarySwapChain == nullptr)
    {
        return {};
    }

    const std::uint64_t requestId = mNextPresentationReadbackRequestId++;
    mPendingPresentationReadbackRequests.emplace(requestId, requestId);
    return GpuPresentationReadbackRequest{requestId};
}

bool GpuDeviceImpl::tryGetPresentationReadback(GpuPresentationReadbackRequest request,
                                               GpuPresentationReadbackEvent &outEvent)
{
    outEvent = {};
    if (request.id == 0)
    {
        return false;
    }

    const auto it = mCompletedPresentationReadbacks.find(request.id);
    if (it == mCompletedPresentationReadbacks.end())
    {
        return false;
    }

    outEvent = std::move(it->second);
    mCompletedPresentationReadbacks.erase(it);
    return true;
}

const std::string &GpuDeviceImpl::shaderSourceDirectory() const
{
    return mDesc.shaderDirectory;
}

bool GpuDeviceImpl::initializeVulkan()
{
    Diligent::IEngineFactoryVk *factoryVk = Diligent::LoadAndGetEngineFactoryVk();
    if (factoryVk == nullptr)
    {
        return false;
    }

    auto createDeviceContexts = [&](bool requestDedicatedPhysicsContext)
    {
        mRenderDevice     = nullptr;
        mImmediateContext = nullptr;
        mPhysicsContext   = nullptr;

        Diligent::EngineVkCreateInfo engineCreateInfo{};
        engineCreateInfo.EnableValidation =
            static_cast<Diligent::Bool>(mDesc.enableValidation ? 1 : 0);

        std::array<Diligent::IDeviceContext *, 2> contexts = {nullptr, nullptr};
        if (requestDedicatedPhysicsContext)
        {
            static constexpr Diligent::ImmediateContextCreateInfo kContextInfo[2] = {
                Diligent::ImmediateContextCreateInfo{"CRESSimNeo.GraphicsContext", 0u},
                Diligent::ImmediateContextCreateInfo{"CRESSimNeo.PhysicsContext", 0u},
            };
            engineCreateInfo.pImmediateContextInfo = kContextInfo;
            engineCreateInfo.NumImmediateContexts  = 2u;
            factoryVk->CreateDeviceAndContextsVk(engineCreateInfo, &mRenderDevice, contexts.data());
        }
        else
        {
            factoryVk->CreateDeviceAndContextsVk(engineCreateInfo, &mRenderDevice, contexts.data());
        }

        if (mRenderDevice == nullptr || contexts[0] == nullptr)
        {
            return false;
        }

        mImmediateContext = contexts[0];
        if (requestDedicatedPhysicsContext && contexts[1] != nullptr)
        {
            mPhysicsContext = contexts[1];
        }
        else
        {
            mPhysicsContext = mImmediateContext;
        }
        return true;
    };

    if (!createDeviceContexts(true))
    {
        CRESSIM_LOG_WARNING("GpuDeviceImpl: failed to create dedicated physics context; falling "
                            "back to shared context.");
        if (!createDeviceContexts(false))
        {
            return false;
        }
    }

    if (mPhysicsContext == mImmediateContext)
    {
        CRESSIM_LOG_WARNING("GpuDeviceImpl: physics context is shared with graphics context.");
    }

    const auto graphicsDesc = mImmediateContext->GetDesc();
    const auto physicsDesc  = mPhysicsContext->GetDesc();
    mGraphicsContextId      = graphicsDesc.ContextId;
    mPhysicsContextId       = physicsDesc.ContextId;
    mGraphicsQueueType      = graphicsDesc.QueueType;
    mPhysicsQueueType       = physicsDesc.QueueType;

    mBackend = GpuBackend::Vulkan;

    if (mRenderDevice != nullptr)
    {
        Diligent::FenceDesc readbackFenceDesc{};
        readbackFenceDesc.Name = "CRESSimNeo.PresentationReadbackFence";
        readbackFenceDesc.Type = Diligent::FENCE_TYPE_CPU_WAIT_ONLY;
        mRenderDevice->CreateFence(readbackFenceDesc, &mPresentationReadbackFence);
    }
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

    Diligent::IEngineFactoryVk *factoryVk = Diligent::LoadAndGetEngineFactoryVk();
    if (factoryVk == nullptr)
    {
        return false;
    }

    Diligent::NativeWindow window{};
#if PLATFORM_WIN32
    if (mDesc.presentation.nativeWindow == nullptr)
    {
        CRESSIM_LOG_ERROR("GpuDeviceImpl: presentation nativeWindow must be set on Win32.");
        return false;
    }
    window.hWnd = mDesc.presentation.nativeWindow;
#elif PLATFORM_LINUX
    if (mDesc.presentation.nativeWindowId == 0)
    {
        CRESSIM_LOG_ERROR("GpuDeviceImpl: presentation nativeWindowId must be set on Linux.");
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
        CRESSIM_LOG_ERROR(
            "GpuDeviceImpl: presentation native display/connection is missing on Linux.");
        return false;
    }
#elif PLATFORM_MACOS
    if (mDesc.presentation.nativeWindow == nullptr)
    {
        CRESSIM_LOG_ERROR("GpuDeviceImpl: presentation nativeWindow must be set on macOS.");
        return false;
    }
    window.pNSView = mDesc.presentation.nativeWindow;
#else
    CRESSIM_LOG_ERROR("GpuDeviceImpl: presentation is unsupported on this platform.");
    return false;
#endif

    Diligent::SwapChainDesc swapChainDesc{};
    const GpuRenderTargetDesc defaultTargetDesc = effectiveDefaultRenderTargetDesc(mDesc);
    swapChainDesc.Width                         = defaultTargetDesc.width;
    swapChainDesc.Height                        = defaultTargetDesc.height;
    swapChainDesc.DepthBufferFormat             = Diligent::TEX_FORMAT_UNKNOWN;
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
        CRESSIM_LOG_ERROR("GpuDeviceImpl: failed to create primary swapchain.");
        return false;
    }

    return true;
}

} // namespace cressim::neo::gpu
