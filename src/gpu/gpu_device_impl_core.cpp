#include "common/logger.h"
#include "gpu/gpu_device_impl.h"
#include "gpu/gpu_render_target_system_impl.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace cressim::neo::gpu
{

std::unique_ptr<GpuDevice> createGpuDevice()
{
    return std::make_unique<GpuDeviceImpl>();
}

GpuDeviceImpl::~GpuDeviceImpl()
{
    shutdown();
}

namespace
{

struct VulkanDedicatedContextPlan
{
    bool supported                  = false;
    bool hasGraphicsQueue           = false;
    Diligent::Uint8 graphicsQueueId = 0u;
    Diligent::Uint8 physicsQueueId  = 0u;
};

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

VulkanDedicatedContextPlan planDedicatedVulkanContexts(Diligent::IEngineFactoryVk &factoryVk,
                                                       const Diligent::Uint32 adapterId)
{
    VulkanDedicatedContextPlan plan{};

    Diligent::Uint32 adapterCount = 0u;
    factoryVk.EnumerateAdapters(Diligent::Version{}, adapterCount, nullptr);
    if (adapterCount == 0u)
    {
        return plan;
    }

    std::vector<Diligent::GraphicsAdapterInfo> adapters(adapterCount);
    factoryVk.EnumerateAdapters(Diligent::Version{}, adapterCount, adapters.data());
    if (adapters.empty())
    {
        return plan;
    }

    const Diligent::Uint32 resolvedAdapterId =
        adapterId == Diligent::DEFAULT_ADAPTER_ID ? 0u : std::min(adapterId, adapterCount - 1u);
    const Diligent::GraphicsAdapterInfo &adapterInfo = adapters[resolvedAdapterId];
    if (adapterInfo.NumQueues == 0u)
    {
        return plan;
    }

    for (Diligent::Uint32 queueIndex = 0; queueIndex < adapterInfo.NumQueues; ++queueIndex)
    {
        const auto &queueInfo = adapterInfo.Queues[queueIndex];
        if (queueInfo.QueueType == Diligent::COMMAND_QUEUE_TYPE_GRAPHICS &&
            queueInfo.MaxDeviceContexts > 0u)
        {
            plan.graphicsQueueId  = static_cast<Diligent::Uint8>(queueIndex);
            plan.hasGraphicsQueue = true;
            break;
        }
    }

    if (!plan.hasGraphicsQueue)
    {
        return plan;
    }

    const auto canRunPhysics = [](Diligent::COMMAND_QUEUE_TYPE queueType)
    { return (queueType & Diligent::COMMAND_QUEUE_TYPE_COMPUTE) != 0; };

    for (Diligent::Uint32 queueIndex = 0; queueIndex < adapterInfo.NumQueues; ++queueIndex)
    {
        if (queueIndex == plan.graphicsQueueId)
        {
            continue;
        }

        const auto &queueInfo = adapterInfo.Queues[queueIndex];
        if (queueInfo.MaxDeviceContexts > 0u && canRunPhysics(queueInfo.QueueType))
        {
            plan.physicsQueueId = static_cast<Diligent::Uint8>(queueIndex);
            plan.supported      = true;
            return plan;
        }
    }

    const auto &graphicsQueueInfo = adapterInfo.Queues[plan.graphicsQueueId];
    if (graphicsQueueInfo.MaxDeviceContexts >= 2u && canRunPhysics(graphicsQueueInfo.QueueType))
    {
        plan.physicsQueueId = plan.graphicsQueueId;
        plan.supported      = true;
    }

    return plan;
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

    if (!mShaderCache.initialize(mRenderDevice))
    {
        shutdown();
        return false;
    }

    if (mDesc.presentation.enabled && !createPrimarySwapChain())
    {
        shutdown();
        return false;
    }

    mRenderTargetSystem = std::make_unique<GpuRenderTargetSystemImpl>();
    if (!mRenderTargetSystem->initialize(mBackend == GpuBackend::Vulkan, mRenderDevice,
                                         mGraphicsContext))
    {
        shutdown();
        return false;
    }

    mInitialized = true;
    return true;
}

void GpuDeviceImpl::shutdown()
{
    if (mRenderTargetSystem != nullptr)
    {
        mRenderTargetSystem->shutdown();
        mRenderTargetSystem.reset();
    }

    if (mGraphicsContext != nullptr)
    {
        mGraphicsContext->SetRenderTargets(0, nullptr, nullptr,
                                           Diligent::RESOURCE_STATE_TRANSITION_MODE_NONE);
        mGraphicsContext->Flush();
        mGraphicsContext->FinishFrame();
    }
    if (mPhysicsContext != nullptr && mPhysicsContext != mGraphicsContext)
    {
        mPhysicsContext->Flush();
        mPhysicsContext->FinishFrame();
    }

    mShaderCache.shutdown();

    mGraphicsContext                    = nullptr;
    mPhysicsContext                     = nullptr;
    mRenderDevice                       = nullptr;
    mPrimarySwapChain                   = nullptr;
    mPresentationReadbackFence          = nullptr;
    mPhysicsToGraphicsFence             = nullptr;
    mGraphicsContextId                  = 0;
    mPhysicsContextId                   = 0;
    mGraphicsQueueType                  = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mPhysicsQueueType                   = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mFrameActive                        = false;
    mNextPresentationReadbackRequestId  = 1;
    mNextPresentationReadbackFenceValue = 1;
    mNextPhysicsToGraphicsFenceValue    = 1;
    mPendingPresentationReadbackRequests.clear();
    mPendingPresentationReadbackCopies.clear();
    mCompletedPresentationReadbacks.clear();

    mBackend     = GpuBackend::Null;
    mInitialized = false;
}

GpuRenderTargetSystem &GpuDeviceImpl::renderTargetSystem()
{
    return *mRenderTargetSystem;
}

GpuBackend GpuDeviceImpl::backend() const
{
    return mBackend;
}

bool GpuDeviceImpl::tryGetGraphicsBackendContext(GpuGraphicsBackendContext &outContext)
{
    outContext = GpuGraphicsBackendContext{};

    if (!mInitialized || mBackend != GpuBackend::Vulkan || mRenderDevice == nullptr ||
        mGraphicsContext == nullptr)
    {
        return false;
    }

    outContext.renderDevice     = mRenderDevice;
    outContext.graphicsContext  = mGraphicsContext;
    outContext.primarySwapChain = mPrimarySwapChain;
    outContext.contextId        = mGraphicsContextId;
    if (mRenderTargetSystem != nullptr)
    {
        mRenderTargetSystem->fillBackendContextState(outContext);
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
    return true;
}

bool GpuDeviceImpl::waitForPhysicsOnGraphics()
{
    if (!mInitialized || mBackend != GpuBackend::Vulkan || mGraphicsContext == nullptr ||
        mPhysicsContext == nullptr)
    {
        return false;
    }

    if (mGraphicsContext == mPhysicsContext || mGraphicsContextId == mPhysicsContextId)
    {
        return true;
    }

    if (mPhysicsToGraphicsFence == nullptr)
    {
        return false;
    }

    const std::uint64_t fenceValue = mNextPhysicsToGraphicsFenceValue++;
    // Shared pose buffers are written on the physics context and then consumed by
    // subsequent graphics uploads and render passes on the graphics context.
    mPhysicsContext->EnqueueSignal(mPhysicsToGraphicsFence, fenceValue);
    mPhysicsContext->Flush();
    mGraphicsContext->DeviceWaitForFence(mPhysicsToGraphicsFence, fenceValue);
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

bool GpuDeviceImpl::createShader(const Diligent::ShaderCreateInfo &createInfo,
                                 Diligent::IShader **shader)
{
    if (shader == nullptr)
    {
        return false;
    }
    *shader = nullptr;

    if (mShaderCache.createShader(createInfo, shader))
    {
        return true;
    }
    if (mRenderDevice == nullptr)
    {
        return false;
    }

    mRenderDevice->CreateShader(createInfo, shader);
    return *shader != nullptr;
}

bool GpuDeviceImpl::createGraphicsPipelineState(
    const Diligent::GraphicsPipelineStateCreateInfo &createInfo,
    Diligent::IPipelineState **pipelineState)
{
    if (pipelineState == nullptr)
    {
        return false;
    }
    *pipelineState = nullptr;

    if (mShaderCache.createGraphicsPipelineState(createInfo, pipelineState))
    {
        return true;
    }
    if (mRenderDevice == nullptr)
    {
        return false;
    }

    mRenderDevice->CreateGraphicsPipelineState(createInfo, pipelineState);
    return *pipelineState != nullptr;
}

bool GpuDeviceImpl::createComputePipelineState(
    const Diligent::ComputePipelineStateCreateInfo &createInfo,
    Diligent::IPipelineState **pipelineState)
{
    if (pipelineState == nullptr)
    {
        return false;
    }
    *pipelineState = nullptr;

    if (mShaderCache.createComputePipelineState(createInfo, pipelineState))
    {
        return true;
    }
    if (mRenderDevice == nullptr)
    {
        return false;
    }

    mRenderDevice->CreateComputePipelineState(createInfo, pipelineState);
    return *pipelineState != nullptr;
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
        mRenderDevice    = nullptr;
        mGraphicsContext = nullptr;
        mPhysicsContext  = nullptr;

        Diligent::EngineVkCreateInfo engineCreateInfo{};
        engineCreateInfo.EnableValidation =
            static_cast<Diligent::Bool>(mDesc.enableValidation ? 1 : 0);

        const VulkanDedicatedContextPlan dedicatedContextPlan =
            planDedicatedVulkanContexts(*factoryVk, engineCreateInfo.AdapterId);

        std::array<Diligent::IDeviceContext *, 2> contexts = {nullptr, nullptr};
        if (requestDedicatedPhysicsContext && dedicatedContextPlan.supported)
        {
            const std::array<Diligent::ImmediateContextCreateInfo, 2> kContextInfo = {
                Diligent::ImmediateContextCreateInfo{"CRESSimNeo.GraphicsContext",
                                                     dedicatedContextPlan.graphicsQueueId},
                Diligent::ImmediateContextCreateInfo{"CRESSimNeo.PhysicsContext",
                                                     dedicatedContextPlan.physicsQueueId},
            };
            engineCreateInfo.pImmediateContextInfo = kContextInfo.data();
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

        mGraphicsContext = contexts[0];
        if (requestDedicatedPhysicsContext && contexts[1] != nullptr)
        {
            mPhysicsContext = contexts[1];
        }
        else
        {
            mPhysicsContext = mGraphicsContext;
        }
        return true;
    };

    if (!createDeviceContexts(true))
    {
        CRESSIM_LOG_WARNING(
            "failed to create dedicated physics context; falling back to shared context.");
        if (!createDeviceContexts(false))
        {
            return false;
        }
    }

    if (mPhysicsContext == mGraphicsContext)
    {
        CRESSIM_LOG_WARNING("physics context is shared with graphics context.");
    }

    const auto graphicsDesc = mGraphicsContext->GetDesc();
    const auto physicsDesc  = mPhysicsContext->GetDesc();
    mGraphicsContextId      = graphicsDesc.ContextId;
    mPhysicsContextId       = physicsDesc.ContextId;
    mGraphicsQueueType      = graphicsDesc.QueueType;
    mPhysicsQueueType       = physicsDesc.QueueType;

    mBackend = GpuBackend::Vulkan;

    if (mRenderDevice != nullptr)
    {
        Diligent::FenceDesc physicsToGraphicsFenceDesc{};
        physicsToGraphicsFenceDesc.Name = "CRESSimNeo.PhysicsToGraphicsFence";
        physicsToGraphicsFenceDesc.Type = Diligent::FENCE_TYPE_GENERAL;
        mRenderDevice->CreateFence(physicsToGraphicsFenceDesc, &mPhysicsToGraphicsFence);
        if (mPhysicsToGraphicsFence == nullptr)
        {
            CRESSIM_LOG_ERROR("failed to create physics-to-graphics fence.");
            return false;
        }

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
    if (mBackend != GpuBackend::Vulkan || mRenderDevice == nullptr || mGraphicsContext == nullptr)
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
        CRESSIM_LOG_ERROR("presentation nativeWindow must be set on Win32.");
        return false;
    }
    window.hWnd = mDesc.presentation.nativeWindow;
#elif PLATFORM_LINUX
    if (mDesc.presentation.nativeWindowId == 0)
    {
        CRESSIM_LOG_ERROR("presentation nativeWindowId must be set on Linux.");
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
        CRESSIM_LOG_ERROR("presentation native display/connection is missing on Linux.");
        return false;
    }
#elif PLATFORM_MACOS
    if (mDesc.presentation.nativeWindow == nullptr)
    {
        CRESSIM_LOG_ERROR("presentation nativeWindow must be set on macOS.");
        return false;
    }
    window.pNSView = mDesc.presentation.nativeWindow;
#else
    CRESSIM_LOG_ERROR("presentation is unsupported on this platform.");
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

    factoryVk->CreateSwapChainVk(mRenderDevice, mGraphicsContext, swapChainDesc, window,
                                 &mPrimarySwapChain);
    if (mPrimarySwapChain == nullptr)
    {
        CRESSIM_LOG_ERROR("failed to create primary swapchain.");
        return false;
    }

    return true;
}

} // namespace cressim::neo::gpu
