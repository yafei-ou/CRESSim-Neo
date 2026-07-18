#include "common/logger.h"
#include "gpu/gpu_device_impl.h"
#include "gpu/gpu_render_target_system_impl.h"

#include <vulkan/vulkan.h>

#if PLATFORM_WIN32
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineD3D12/interface/EngineFactoryD3D12.h"
#endif
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsTools/interface/ShaderMacroHelper.hpp"
#include "DiligentEngine/DiligentCore/Graphics/ShaderTools/include/DXCompiler.hpp"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
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

#if PLATFORM_WIN32
constexpr const char *kVkKhrExternalMemoryWin32ExtensionName    = "VK_KHR_external_memory_win32";
constexpr const char *kVkKhrExternalSemaphoreWin32ExtensionName = "VK_KHR_external_semaphore_win32";
#endif

struct VulkanDedicatedContextPlan
{
    bool supported                  = false;
    bool hasGraphicsQueue           = false;
    Diligent::Uint8 graphicsQueueId = 0u;
    Diligent::Uint8 physicsQueueId  = 0u;
};

const char *shaderCompilerModeToString(VulkanShaderCompilerMode mode)
{
    switch (mode)
    {
    case VulkanShaderCompilerMode::ForceDefault:
        return "force-default";
    case VulkanShaderCompilerMode::ForceDXC:
        return "force-dxc";
    case VulkanShaderCompilerMode::Auto:
    default:
        return "auto";
    }
}

const char *adapterTypeToString(Diligent::ADAPTER_TYPE type)
{
    switch (type)
    {
    case Diligent::ADAPTER_TYPE_SOFTWARE:
        return "software";
    case Diligent::ADAPTER_TYPE_INTEGRATED:
        return "integrated";
    case Diligent::ADAPTER_TYPE_DISCRETE:
        return "discrete";
    default:
        return "unknown";
    }
}

void logDedicatedContextPlan(const VulkanDedicatedContextPlan &plan,
                             const Diligent::GraphicsAdapterInfo *adapterInfo,
                             Diligent::Uint32 adapterId)
{
    if (!plan.hasGraphicsQueue)
    {
        return;
    }

    if (!plan.supported)
    {
        return;
    }

    const char *adapterDescription = adapterInfo != nullptr ? adapterInfo->Description : "unknown";
    CRESSIM_LOG_INFO(
        "Vulkan dedicated-context plan: adapter=", adapterId, " ('", adapterDescription,
        "', type=", adapterInfo != nullptr ? adapterTypeToString(adapterInfo->Type) : "unknown",
        "), graphicsQueue=", static_cast<Diligent::Uint32>(plan.graphicsQueueId),
        ", physicsQueue=", static_cast<Diligent::Uint32>(plan.physicsQueueId), " (",
        plan.graphicsQueueId == plan.physicsQueueId ? "shared queue, separate contexts"
                                                    : "separate queues",
        ").");
}

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

bool isDiligentGraphicsBackend(GpuBackend backend) noexcept
{
    return backend == GpuBackend::D3D12 || backend == GpuBackend::Vulkan;
}

bool probeNativePhysicsFloatAtomicShader(Diligent::IRenderDevice *renderDevice,
                                         VulkanShaderCompilerMode shaderCompilerMode)
{
    if (renderDevice == nullptr)
    {
        CRESSIM_LOG_WARNING(
            "Skipping native physics float-atomics shader probe: render device is null.");
        return false;
    }
    static constexpr const char *kNativeFloatAtomicProbeShader = R"hlsl(
        RWStructuredBuffer<float4> g_Output;

        static const uint kCressimSpirvScopeDevice = 1u;
        static const uint kCressimSpirvMemorySemanticsAcquireReleaseUniform = 0x48u;

        [[vk::ext_capability(6033)]]
        [[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
        [[vk::ext_instruction(6035)]]
        float CressimAtomicFloatAddNative([[vk::ext_reference]] float pointer, uint scope,
                                        uint semantics, float value);

        [numthreads(1, 1, 1)]
        void main(uint3 dispatchThreadID : SV_DispatchThreadID)
        {
            CressimAtomicFloatAddNative(g_Output[0].x, kCressimSpirvScopeDevice,
                                        kCressimSpirvMemorySemanticsAcquireReleaseUniform, 1.0);
        }
        )hlsl";

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.Desc.ShaderType                 = Diligent::SHADER_TYPE_COMPUTE;
    shaderCreateInfo.Desc.Name                       = "CRESSimNeo.NativeFloatAtomicsProbe";
    shaderCreateInfo.Source                          = kNativeFloatAtomicProbeShader;
    shaderCreateInfo.ShaderCompiler                  = Diligent::SHADER_COMPILER_DXC;

    CRESSIM_LOG_INFO(
        "Probing native physics float atomics with a built-in compute shader, compiler=",
        shaderCompilerModeToString(shaderCompilerMode), ".");

    Diligent::RefCntAutoPtr<Diligent::IShader> probeShader;
    renderDevice->CreateShader(shaderCreateInfo, &probeShader);
    if (probeShader == nullptr)
    {
        CRESSIM_LOG_WARNING(
            "Native physics float-atomics shader probe failed; using CAS fallback on this "
            "device.");
        return false;
    }

    CRESSIM_LOG_INFO("Native physics float-atomics shader probe succeeded.");
    return true;
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
        if ((queueInfo.QueueType & Diligent::COMMAND_QUEUE_TYPE_GRAPHICS) ==
                Diligent::COMMAND_QUEUE_TYPE_GRAPHICS &&
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

    const auto &graphicsQueueInfo = adapterInfo.Queues[plan.graphicsQueueId];
    if (graphicsQueueInfo.MaxDeviceContexts >= 2u && canRunPhysics(graphicsQueueInfo.QueueType))
    {
        plan.physicsQueueId = plan.graphicsQueueId;
        plan.supported      = true;
        return plan;
    }

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

    return plan;
}

} // namespace

bool GpuDeviceImpl::initialize(const GpuDeviceDesc &desc)
{
    shutdown();

    mDesc = desc;

    try
    {
        switch (mDesc.preferredBackend)
        {
        case GpuBackend::Null:
            if (!initializeNull())
            {
                shutdown();
                return false;
            }
            break;
        case GpuBackend::D3D12:
            if (!initializeD3D12())
            {
                shutdown();
                return false;
            }
            break;
        case GpuBackend::Vulkan:
            if (!initializeVulkan())
            {
                shutdown();
                return false;
            }
            break;
        default:
            return false;
        }

        if (mRenderDevice != nullptr && !mShaderCache.initialize(mRenderDevice))
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
        if (!mRenderTargetSystem->initialize(mBackend, mRenderDevice, mGraphicsContext))
        {
            shutdown();
            return false;
        }
    }
    catch (const std::exception &exception)
    {
        CRESSIM_LOG_ERROR("GpuDevice initialization failed for backend ",
                          static_cast<std::uint32_t>(mDesc.preferredBackend),
                          " with exception: ", exception.what());
        shutdown();
        return false;
    }

    mInitialized = true;
    return true;
}

void GpuDeviceImpl::shutdown()
{
    mFrameActive = false;

    mPendingPresentationReadbackRequests.clear();
    mPendingPresentationReadbackCopies.clear();
    mCompletedPresentationReadbacks.clear();

    mNextPresentationReadbackRequestId  = 1;
    mNextPresentationReadbackFenceValue = 1;
    mNextPhysicsToGraphicsFenceValue    = 1;
    mNextGraphicsToPhysicsFenceValue    = 1;

    // Release swapchain- and fence-owned device children before tearing down the contexts/device.
    mPrimarySwapChain          = nullptr;
    mPresentationReadbackFence = nullptr;
    mPhysicsToGraphicsFence    = nullptr;
    mGraphicsToPhysicsFence    = nullptr;

    if (mRenderTargetSystem != nullptr)
    {
        mRenderTargetSystem->shutdown();
        mRenderTargetSystem.reset();
    }

    mShaderCache.shutdown();

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

    if (mRenderDevice != nullptr)
    {
        // Diligent's Vulkan command queues can retain hundreds of pending-release device objects
        // after the contexts are finished. Drain those queues before releasing our last device refs
        // so the render device destructor can actually run.
        mRenderDevice->IdleGPU();
        mRenderDevice->ReleaseStaleResources(true);
    }

    mGraphicsContext = nullptr;
    mPhysicsContext  = nullptr;

    mRenderDevice                      = nullptr;
    mGraphicsContextId                 = 0;
    mPhysicsContextId                  = 0;
    mGraphicsQueueType                 = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mPhysicsQueueType                  = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mSupportsNativePhysicsFloatAtomics = false;

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

bool GpuDeviceImpl::initializeNull()
{
    if (mDesc.presentation.enabled)
    {
        CRESSIM_LOG_ERROR("Null GPU backend does not support presentation.");
        return false;
    }

    mBackend = GpuBackend::Null;
    return true;
}

bool GpuDeviceImpl::tryGetGraphicsBackendContext(GpuGraphicsBackendContext &outContext)
{
    outContext = GpuGraphicsBackendContext{};

    if (!mInitialized || !isDiligentGraphicsBackend(mBackend) || mRenderDevice == nullptr ||
        mGraphicsContext == nullptr)
    {
        return false;
    }

    outContext.renderDevice     = mRenderDevice;
    outContext.graphicsContext  = mGraphicsContext;
    outContext.primarySwapChain = mPrimarySwapChain;
    outContext.contextId        = mGraphicsContextId;
    outContext.queueType        = mGraphicsQueueType;
    if (mRenderTargetSystem != nullptr)
    {
        mRenderTargetSystem->fillBackendContextState(outContext);
    }
    return true;
}

bool GpuDeviceImpl::tryGetPhysicsBackendContext(GpuComputeBackendContext &outContext)
{
    outContext = GpuComputeBackendContext{};

    if (!mInitialized || !isDiligentGraphicsBackend(mBackend) || mRenderDevice == nullptr ||
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
    if (!mInitialized || !isDiligentGraphicsBackend(mBackend) || mGraphicsContext == nullptr ||
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

bool GpuDeviceImpl::waitForGraphicsOnPhysics()
{
    if (!mInitialized || !isDiligentGraphicsBackend(mBackend) || mGraphicsContext == nullptr ||
        mPhysicsContext == nullptr)
    {
        return false;
    }

    if (mGraphicsContext == mPhysicsContext || mGraphicsContextId == mPhysicsContextId)
    {
        return true;
    }

    if (mGraphicsToPhysicsFence == nullptr)
    {
        return false;
    }

    const std::uint64_t fenceValue = mNextGraphicsToPhysicsFenceValue++;
    // Shared pose buffers are consumed on the graphics context and then updated again by the
    // physics context on the following frame.
    mGraphicsContext->EnqueueSignal(mGraphicsToPhysicsFence, fenceValue);
    mGraphicsContext->Flush();
    mPhysicsContext->DeviceWaitForFence(mGraphicsToPhysicsFence, fenceValue);
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
    if (!mInitialized || !isDiligentGraphicsBackend(mBackend) || mPrimarySwapChain == nullptr)
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
    if (!mInitialized || !isDiligentGraphicsBackend(mBackend) || mPrimarySwapChain == nullptr)
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

bool GpuDeviceImpl::supportsNativePhysicsFloatAtomics() const
{
    return mSupportsNativePhysicsFloatAtomics;
}

const std::string &GpuDeviceImpl::shaderSourceDirectory() const
{
    return mDesc.shaderDirectory;
}

ShaderSourceConfig GpuDeviceImpl::shaderSourceConfig() const
{
    ShaderSourceConfig config{};
    config.sourceDirectory    = mDesc.shaderDirectory;
    config.includeDirectories = mDesc.shaderIncludeDirectories;
    return config;
}

bool GpuDeviceImpl::createShader(const Diligent::ShaderCreateInfo &createInfo,
                                 Diligent::IShader **shader)
{
    if (shader == nullptr)
    {
        return false;
    }
    *shader = nullptr;

    Diligent::ShaderCreateInfo shaderCreateInfo = createInfo;
    Diligent::ShaderMacroHelper shaderMacros;
    const bool isPhysicsShader = shaderCreateInfo.FilePath != nullptr &&
                                 std::strstr(shaderCreateInfo.FilePath, "physics/") != nullptr;
    if (isPhysicsShader)
    {
        shaderMacros += shaderCreateInfo.Macros;
        shaderMacros.Add("CRESSIM_NATIVE_FLOAT_BUFFER_ATOMICS", mSupportsNativePhysicsFloatAtomics);
        shaderCreateInfo.Macros = shaderMacros;
    }
    if (mBackend == GpuBackend::Vulkan)
    {
        switch (mDesc.vulkanShaderCompilerMode)
        {
        case VulkanShaderCompilerMode::ForceDefault:
            shaderCreateInfo.ShaderCompiler = Diligent::SHADER_COMPILER_DEFAULT;
            break;
        case VulkanShaderCompilerMode::ForceDXC:
            shaderCreateInfo.ShaderCompiler = Diligent::SHADER_COMPILER_DXC;
            break;
        case VulkanShaderCompilerMode::Auto:
        default:
            break;
        }

        if (isPhysicsShader && mSupportsNativePhysicsFloatAtomics &&
            mDesc.vulkanShaderCompilerMode == VulkanShaderCompilerMode::Auto)
        {
            shaderCreateInfo.ShaderCompiler = Diligent::SHADER_COMPILER_DXC;
        }
    }
    else if (mBackend == GpuBackend::D3D12)
    {
        shaderCreateInfo.ShaderCompiler = Diligent::SHADER_COMPILER_DXC;
    }

    if (mShaderCache.createShader(shaderCreateInfo, shader))
    {
        return true;
    }
    if (mRenderDevice == nullptr)
    {
        return false;
    }

    mRenderDevice->CreateShader(shaderCreateInfo, shader);
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

    bool cacheAttempted = false;
    if (mShaderCache.createGraphicsPipelineState(createInfo, pipelineState, &cacheAttempted))
    {
        return true;
    }
    if (mRenderDevice == nullptr)
    {
        return false;
    }
    if (cacheAttempted)
    {
        const char *pipelineName =
            createInfo.PSODesc.Name != nullptr ? createInfo.PSODesc.Name : "<unnamed>";
        CRESSIM_LOG_WARNING("cached graphics PSO creation failed for '", pipelineName,
                            "'; falling back to direct device creation.");
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

    const VulkanShaderCompilerMode shaderCompilerMode = mDesc.vulkanShaderCompilerMode;
    const Diligent::Version &supportedVkVersion       = factoryVk->GetVulkanVersion();
    const Diligent::Uint32 dxCompilerVkVersion =
        VK_MAKE_VERSION(supportedVkVersion.Major, supportedVkVersion.Minor, 0u);
    std::unique_ptr<Diligent::IDXCompiler> vulkanDxCompiler = Diligent::CreateDXCompiler(
        Diligent::DXCompilerTarget::Vulkan, dxCompilerVkVersion, nullptr);
    const bool dxcLoaded = vulkanDxCompiler != nullptr && vulkanDxCompiler->IsLoaded();
    const bool canCompileNativePhysicsShaders =
        dxcLoaded && shaderCompilerMode != VulkanShaderCompilerMode::ForceDefault;

    if (shaderCompilerMode == VulkanShaderCompilerMode::ForceDXC && !dxcLoaded)
    {
        CRESSIM_LOG_ERROR(
            "Vulkan shader compiler mode is set to force DXC, but Diligent could not load the "
            "Vulkan DXC compiler.");
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
        std::array<const char *, 2> instanceExtensions{};
        Diligent::Uint32 requestedInstanceExtensionCount = 0u;
        std::array<const char *, 5> deviceExtensions{};
        Diligent::Uint32 requestedDeviceExtensionCount = 0u;
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shaderAtomicFloatFeatures{};
        deviceExtensions[requestedDeviceExtensionCount++] =
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;

#if CRESSIM_NEO_HAS_CUDA_INTEROP
        engineCreateInfo.Features.NativeFence = Diligent::DEVICE_FEATURE_STATE_ENABLED;
        instanceExtensions[requestedInstanceExtensionCount++] =
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME;
        instanceExtensions[requestedInstanceExtensionCount++] =
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME;
        deviceExtensions[requestedDeviceExtensionCount++] = VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME;
#if PLATFORM_WIN32
        deviceExtensions[requestedDeviceExtensionCount++] = kVkKhrExternalMemoryWin32ExtensionName;
        deviceExtensions[requestedDeviceExtensionCount++] =
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME;
        deviceExtensions[requestedDeviceExtensionCount++] =
            kVkKhrExternalSemaphoreWin32ExtensionName;
#elif PLATFORM_LINUX
        deviceExtensions[requestedDeviceExtensionCount++] =
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
        deviceExtensions[requestedDeviceExtensionCount++] =
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME;
        deviceExtensions[requestedDeviceExtensionCount++] =
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif
#endif

        engineCreateInfo.InstanceExtensionCount = requestedInstanceExtensionCount;
        engineCreateInfo.ppInstanceExtensionNames =
            requestedInstanceExtensionCount > 0u ? instanceExtensions.data() : nullptr;
        engineCreateInfo.DeviceExtensionCount   = requestedDeviceExtensionCount;
        engineCreateInfo.ppDeviceExtensionNames = deviceExtensions.data();

        shaderAtomicFloatFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
        engineCreateInfo.pDeviceExtensionFeatures              = &shaderAtomicFloatFeatures;

        CRESSIM_LOG_INFO("Requesting Vulkan native float atomics via ",
                         VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
                         " with shaderBufferFloat32AtomicAdd=VK_TRUE.");
#if CRESSIM_NEO_HAS_CUDA_INTEROP
        CRESSIM_LOG_INFO("Requesting Vulkan CUDA interop instance extensions: ",
                         VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME, ", ",
                         VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME, ".");
#if PLATFORM_WIN32
        CRESSIM_LOG_INFO("Requesting Vulkan CUDA interop device extensions: ",
                         VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, ", ",
                         kVkKhrExternalMemoryWin32ExtensionName, ", ",
                         VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, ", ",
                         kVkKhrExternalSemaphoreWin32ExtensionName, ".");
#elif PLATFORM_LINUX
        CRESSIM_LOG_INFO("Requesting Vulkan CUDA interop device extensions: ",
                         VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, ", ",
                         VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, ", ",
                         VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, ", ",
                         VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, ".");
#endif
#endif

        const VulkanDedicatedContextPlan dedicatedContextPlan =
            planDedicatedVulkanContexts(*factoryVk, engineCreateInfo.AdapterId);
        Diligent::Uint32 adapterCount = 0u;
        factoryVk->EnumerateAdapters(Diligent::Version{}, adapterCount, nullptr);
        std::vector<Diligent::GraphicsAdapterInfo> adapters(adapterCount);
        if (adapterCount > 0u)
        {
            factoryVk->EnumerateAdapters(Diligent::Version{}, adapterCount, adapters.data());
        }
        const Diligent::Uint32 resolvedAdapterId =
            adapterCount == 0u ? 0u
                               : (engineCreateInfo.AdapterId == Diligent::DEFAULT_ADAPTER_ID
                                      ? 0u
                                      : std::min(engineCreateInfo.AdapterId, adapterCount - 1u));
        const Diligent::GraphicsAdapterInfo *adapterInfo =
            resolvedAdapterId < adapters.size() ? &adapters[resolvedAdapterId] : nullptr;

        std::array<Diligent::IDeviceContext *, 2> contexts = {nullptr, nullptr};
        if (requestDedicatedPhysicsContext && dedicatedContextPlan.supported)
        {
            logDedicatedContextPlan(dedicatedContextPlan, adapterInfo, resolvedAdapterId);
            const std::array<Diligent::ImmediateContextCreateInfo, 2> kContextInfo = {
                Diligent::ImmediateContextCreateInfo{"CRESSimNeo.GraphicsContext",
                                                     dedicatedContextPlan.graphicsQueueId},
                Diligent::ImmediateContextCreateInfo{"CRESSimNeo.PhysicsContext",
                                                     dedicatedContextPlan.physicsQueueId},
            };
            CRESSIM_LOG_INFO("Attempting Vulkan device creation with two immediate contexts.");
            engineCreateInfo.pImmediateContextInfo = kContextInfo.data();
            engineCreateInfo.NumImmediateContexts  = 2u;
            factoryVk->CreateDeviceAndContextsVk(engineCreateInfo, &mRenderDevice, contexts.data());
        }
        else
        {
            if (requestDedicatedPhysicsContext)
            {
                logDedicatedContextPlan(dedicatedContextPlan, adapterInfo, resolvedAdapterId);
            }
            factoryVk->CreateDeviceAndContextsVk(engineCreateInfo, &mRenderDevice, contexts.data());
        }

        if (mRenderDevice == nullptr || contexts[0] == nullptr)
        {
            CRESSIM_LOG_WARNING("Vulkan device creation failed. dedicatedPhysicsContext=",
                                requestDedicatedPhysicsContext ? "requested" : "disabled", ".");
            return false;
        }

        // CreateDeviceAndContextsVk() returns owned context pointers. Adopt them without AddRef()
        // so we do not leak an extra strong reference per immediate context.
        mGraphicsContext.Attach(contexts[0]);
        if (requestDedicatedPhysicsContext && contexts[1] != nullptr)
        {
            mPhysicsContext.Attach(contexts[1]);
        }
        else
        {
            mPhysicsContext = mGraphicsContext;
        }
        return true;
    };

    CRESSIM_LOG_INFO(
        "Attempting Vulkan device creation with native physics float atomics requested.");
    if (!createDeviceContexts(true))
    {
        CRESSIM_LOG_WARNING(
            "failed to create dedicated physics context; falling back to shared context.");
        if (!createDeviceContexts(false))
        {
            return false;
        }
    }

    if (canCompileNativePhysicsShaders)
    {
        if (probeNativePhysicsFloatAtomicShader(mRenderDevice, shaderCompilerMode))
        {
            mSupportsNativePhysicsFloatAtomics = true;
        }
    }
    else if (!dxcLoaded)
    {
        CRESSIM_LOG_WARNING(
            "native physics float atomics were requested from Vulkan, but Diligent could not "
            "load the Vulkan DXC compiler; using CAS fallback.");
    }
    else if (shaderCompilerMode == VulkanShaderCompilerMode::ForceDefault)
    {
        CRESSIM_LOG_WARNING(
            "native physics float atomics were requested from Vulkan, but Vulkan shader "
            "compiler mode forces the default compiler; using CAS fallback.");
    }

    if (canCompileNativePhysicsShaders && !mSupportsNativePhysicsFloatAtomics)
    {
        CRESSIM_LOG_WARNING(
            "native physics float atomics were requested, but the created Vulkan device could "
            "not compile the native float-atomics shader; using CAS fallback.");
    }

    CRESSIM_LOG_INFO("Physics float atomics: native=",
                     mSupportsNativePhysicsFloatAtomics ? "enabled" : "disabled",
                     ", shaderCompilerMode=", shaderCompilerModeToString(shaderCompilerMode),
                     ", dxcLoaded=", dxcLoaded ? "yes" : "no", ".");

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

        Diligent::FenceDesc graphicsToPhysicsFenceDesc{};
        graphicsToPhysicsFenceDesc.Name = "CRESSimNeo.GraphicsToPhysicsFence";
        graphicsToPhysicsFenceDesc.Type = Diligent::FENCE_TYPE_GENERAL;
        mRenderDevice->CreateFence(graphicsToPhysicsFenceDesc, &mGraphicsToPhysicsFence);
        if (mGraphicsToPhysicsFence == nullptr)
        {
            CRESSIM_LOG_ERROR("failed to create graphics-to-physics fence.");
            return false;
        }

        Diligent::FenceDesc readbackFenceDesc{};
        readbackFenceDesc.Name = "CRESSimNeo.PresentationReadbackFence";
        readbackFenceDesc.Type = Diligent::FENCE_TYPE_CPU_WAIT_ONLY;
        mRenderDevice->CreateFence(readbackFenceDesc, &mPresentationReadbackFence);
    }
    return true;
}

#if PLATFORM_WIN32
bool GpuDeviceImpl::initializeD3D12()
{
    Diligent::IEngineFactoryD3D12 *factoryD3D12 = Diligent::LoadAndGetEngineFactoryD3D12();
    if (factoryD3D12 == nullptr)
    {
        return false;
    }

    mRenderDevice    = nullptr;
    mGraphicsContext = nullptr;
    mPhysicsContext  = nullptr;

    Diligent::EngineD3D12CreateInfo engineCreateInfo{};
    engineCreateInfo.EnableValidation = static_cast<Diligent::Bool>(mDesc.enableValidation ? 1 : 0);

    const std::array<Diligent::ImmediateContextCreateInfo, 2> kContextInfo = {
        Diligent::ImmediateContextCreateInfo{"CRESSimNeo.GraphicsContext", 0u},
        Diligent::ImmediateContextCreateInfo{"CRESSimNeo.PhysicsContext", 1u},
    };
    engineCreateInfo.NumImmediateContexts  = static_cast<Diligent::Uint32>(kContextInfo.size());
    engineCreateInfo.pImmediateContextInfo = kContextInfo.data();

    std::array<Diligent::IDeviceContext *, 2> contexts = {nullptr, nullptr};
    factoryD3D12->CreateDeviceAndContextsD3D12(engineCreateInfo, &mRenderDevice, contexts.data());
    if (mRenderDevice == nullptr || contexts[0] == nullptr || contexts[1] == nullptr)
    {
        CRESSIM_LOG_ERROR(
            "D3D12 device creation failed to produce split graphics/physics contexts.");
        return false;
    }

    mGraphicsContext.Attach(contexts[0]);
    mPhysicsContext.Attach(contexts[1]);

    const auto graphicsDesc = mGraphicsContext->GetDesc();
    const auto physicsDesc  = mPhysicsContext->GetDesc();
    if ((graphicsDesc.QueueType & Diligent::COMMAND_QUEUE_TYPE_GRAPHICS) == 0)
    {
        CRESSIM_LOG_ERROR("D3D12 graphics context was not created on a graphics-capable queue.");
        return false;
    }
    if ((physicsDesc.QueueType & Diligent::COMMAND_QUEUE_TYPE_COMPUTE) == 0)
    {
        CRESSIM_LOG_ERROR("D3D12 physics context was not created on a compute-capable queue.");
        return false;
    }
    if (graphicsDesc.ContextId == physicsDesc.ContextId)
    {
        CRESSIM_LOG_ERROR("D3D12 graphics and physics contexts unexpectedly share a context id.");
        return false;
    }

    mGraphicsContextId = graphicsDesc.ContextId;
    mPhysicsContextId  = physicsDesc.ContextId;
    mGraphicsQueueType = graphicsDesc.QueueType;
    mPhysicsQueueType  = physicsDesc.QueueType;
    mBackend           = GpuBackend::D3D12;

    Diligent::FenceDesc physicsToGraphicsFenceDesc{};
    physicsToGraphicsFenceDesc.Name = "CRESSimNeo.PhysicsToGraphicsFence";
    physicsToGraphicsFenceDesc.Type = Diligent::FENCE_TYPE_GENERAL;
    mRenderDevice->CreateFence(physicsToGraphicsFenceDesc, &mPhysicsToGraphicsFence);
    if (mPhysicsToGraphicsFence == nullptr)
    {
        CRESSIM_LOG_ERROR("failed to create D3D12 physics-to-graphics fence.");
        return false;
    }

    Diligent::FenceDesc graphicsToPhysicsFenceDesc{};
    graphicsToPhysicsFenceDesc.Name = "CRESSimNeo.GraphicsToPhysicsFence";
    graphicsToPhysicsFenceDesc.Type = Diligent::FENCE_TYPE_GENERAL;
    mRenderDevice->CreateFence(graphicsToPhysicsFenceDesc, &mGraphicsToPhysicsFence);
    if (mGraphicsToPhysicsFence == nullptr)
    {
        CRESSIM_LOG_ERROR("failed to create D3D12 graphics-to-physics fence.");
        return false;
    }

    Diligent::FenceDesc readbackFenceDesc{};
    readbackFenceDesc.Name = "CRESSimNeo.PresentationReadbackFence";
    readbackFenceDesc.Type = Diligent::FENCE_TYPE_CPU_WAIT_ONLY;
    mRenderDevice->CreateFence(readbackFenceDesc, &mPresentationReadbackFence);
    return true;
}
#else
bool GpuDeviceImpl::initializeD3D12()
{
    CRESSIM_LOG_ERROR("D3D12 backend is only supported on Win32 builds.");
    return false;
}
#endif

bool GpuDeviceImpl::createPrimarySwapChain()
{
    if (!mDesc.presentation.enabled)
    {
        return true;
    }
    if (!isDiligentGraphicsBackend(mBackend) || mRenderDevice == nullptr ||
        mGraphicsContext == nullptr)
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

    if (mBackend == GpuBackend::Vulkan)
    {
        Diligent::IEngineFactoryVk *factoryVk = Diligent::LoadAndGetEngineFactoryVk();
        if (factoryVk == nullptr)
        {
            return false;
        }
        factoryVk->CreateSwapChainVk(mRenderDevice, mGraphicsContext, swapChainDesc, window,
                                     &mPrimarySwapChain);
    }
#if PLATFORM_WIN32
    else if (mBackend == GpuBackend::D3D12)
    {
        Diligent::IEngineFactoryD3D12 *factoryD3D12 = Diligent::LoadAndGetEngineFactoryD3D12();
        if (factoryD3D12 == nullptr)
        {
            return false;
        }
        factoryD3D12->CreateSwapChainD3D12(mRenderDevice, mGraphicsContext, swapChainDesc,
                                           Diligent::FullScreenModeDesc{}, window,
                                           &mPrimarySwapChain);
    }
#endif
    if (mPrimarySwapChain == nullptr)
    {
        CRESSIM_LOG_ERROR("failed to create primary swapchain.");
        return false;
    }

    return true;
}

} // namespace cressim::neo::gpu
