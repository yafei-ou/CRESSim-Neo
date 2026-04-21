#include "common/logger.h"
#include "gpu/gpu_device_impl.h"
#include "gpu/gpu_render_target_system_impl.h"

#include <vulkan/vulkan.h>
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/RenderDeviceVk.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsTools/interface/ShaderMacroHelper.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"
#include "DiligentEngine/DiligentCore/Platforms/interface/NativeWindow.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <unistd.h>
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

struct VulkanFloatAtomicAddSupport
{
    bool extensionSupported          = false;
    bool bufferFloat32AtomicAdd      = false;

    bool canUseNativePhysicsAtomics() const
    {
        return extensionSupported && bufferFloat32AtomicAdd;
    }
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

const char *physicsFloatAtomicModeToString(PhysicsFloatAtomicMode mode)
{
    switch (mode)
    {
    case PhysicsFloatAtomicMode::ForceCAS:
        return "force-cas";
    case PhysicsFloatAtomicMode::ForceNative:
        return "force-native";
    case PhysicsFloatAtomicMode::Auto:
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

VulkanFloatAtomicAddSupport queryVulkanFloatAtomicAddSupport(Diligent::Uint32 adapterId)
{
    VulkanFloatAtomicAddSupport support{};
    void *vulkanLibrary = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (vulkanLibrary == nullptr)
    {
        return support;
    }

    const auto closeLibrary = [&]()
    {
        if (vulkanLibrary != nullptr)
        {
            dlclose(vulkanLibrary);
            vulkanLibrary = nullptr;
        }
    };

    auto *getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(vulkanLibrary, "vkGetInstanceProcAddr"));
    if (getInstanceProcAddr == nullptr)
    {
        closeLibrary();
        return support;
    }

    auto *enumerateInstanceExtensionProperties =
        reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
            getInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    auto *createInstance = reinterpret_cast<PFN_vkCreateInstance>(
        getInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (enumerateInstanceExtensionProperties == nullptr || createInstance == nullptr)
    {
        closeLibrary();
        return support;
    }

    std::uint32_t instanceExtensionCount = 0u;
    if (enumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr) !=
        VK_SUCCESS)
    {
        closeLibrary();
        return support;
    }

    std::vector<VkExtensionProperties> instanceExtensions(instanceExtensionCount);
    if (instanceExtensionCount > 0u &&
        enumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount,
                                             instanceExtensions.data()) != VK_SUCCESS)
    {
        closeLibrary();
        return support;
    }

    const bool hasProperties2Extension = std::any_of(
        instanceExtensions.begin(), instanceExtensions.end(), [](const VkExtensionProperties &ext)
        { return std::strcmp(ext.extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0; });

    std::array<const char *, 1> enabledInstanceExtensions{};
    std::uint32_t enabledInstanceExtensionCount = 0u;
    if (hasProperties2Extension)
    {
        enabledInstanceExtensions[enabledInstanceExtensionCount++] =
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
    }

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName   = "CRESSimNeoFloatAtomicProbe";
    applicationInfo.applicationVersion = 1u;
    applicationInfo.pEngineName        = "CRESSimNeo";
    applicationInfo.engineVersion      = 1u;
    applicationInfo.apiVersion         = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo        = &applicationInfo;
    instanceCreateInfo.enabledExtensionCount   = enabledInstanceExtensionCount;
    instanceCreateInfo.ppEnabledExtensionNames =
        enabledInstanceExtensionCount > 0u ? enabledInstanceExtensions.data() : nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    if (createInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS ||
        instance == VK_NULL_HANDLE)
    {
        closeLibrary();
        return support;
    }

    auto *destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
        getInstanceProcAddr(instance, "vkDestroyInstance"));
    auto *enumeratePhysicalDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
    auto *enumerateDeviceExtensionProperties =
        reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
            getInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties"));
    auto *getPhysicalDeviceFeatures2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            getInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
    auto *getPhysicalDeviceFeatures2KHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
            getInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));

    if (destroyInstance == nullptr || enumeratePhysicalDevices == nullptr ||
        enumerateDeviceExtensionProperties == nullptr ||
        (getPhysicalDeviceFeatures2 == nullptr && getPhysicalDeviceFeatures2KHR == nullptr))
    {
        destroyInstance(instance, nullptr);
        closeLibrary();
        return support;
    }

    std::uint32_t physicalDeviceCount = 0u;
    if (enumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr) != VK_SUCCESS ||
        physicalDeviceCount == 0u)
    {
        destroyInstance(instance, nullptr);
        closeLibrary();
        return support;
    }

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    if (enumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()) !=
        VK_SUCCESS)
    {
        destroyInstance(instance, nullptr);
        closeLibrary();
        return support;
    }

    const std::uint32_t resolvedAdapterId =
        adapterId == Diligent::DEFAULT_ADAPTER_ID ? 0u : std::min(adapterId, physicalDeviceCount - 1u);
    const VkPhysicalDevice physicalDevice = physicalDevices[resolvedAdapterId];

    std::uint32_t deviceExtensionCount = 0u;
    if (enumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtensionCount,
                                           nullptr) != VK_SUCCESS)
    {
        destroyInstance(instance, nullptr);
        closeLibrary();
        return support;
    }

    std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
    if (deviceExtensionCount > 0u &&
        enumerateDeviceExtensionProperties(physicalDevice, nullptr, &deviceExtensionCount,
                                           deviceExtensions.data()) != VK_SUCCESS)
    {
        destroyInstance(instance, nullptr);
        closeLibrary();
        return support;
    }

    support.extensionSupported = std::any_of(
        deviceExtensions.begin(), deviceExtensions.end(), [](const VkExtensionProperties &ext)
        { return std::strcmp(ext.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0; });

    if (support.extensionSupported)
    {
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shaderAtomicFloatFeatures{};
        shaderAtomicFloatFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &shaderAtomicFloatFeatures;

        if (getPhysicalDeviceFeatures2 != nullptr)
        {
            getPhysicalDeviceFeatures2(physicalDevice, &features2);
        }
        else
        {
            getPhysicalDeviceFeatures2KHR(physicalDevice, &features2);
        }

        support.bufferFloat32AtomicAdd =
            shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd == VK_TRUE;
    }

    destroyInstance(instance, nullptr);
    closeLibrary();
    return support;
}

std::filesystem::path getExecutableDirectory()
{
    std::array<char, 4096> executablePath{};
    const ssize_t length = readlink("/proc/self/exe", executablePath.data(), executablePath.size() - 1);
    if (length <= 0)
    {
        return {};
    }

    executablePath[static_cast<std::size_t>(length)] = '\0';
    return std::filesystem::path(executablePath.data()).parent_path();
}

std::string findBundledVulkanDXCompilerPath()
{
    const std::filesystem::path executableDirectory = getExecutableDirectory();
    if (executableDirectory.empty())
    {
        return {};
    }

    const std::filesystem::path bundledDxCompilerPath = executableDirectory / "libdxcompiler.so";
    if (!std::filesystem::exists(bundledDxCompilerPath))
    {
        return {};
    }

    return bundledDxCompilerPath.string();
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
    mGraphicsToPhysicsFence             = nullptr;
    mGraphicsContextId                  = 0;
    mPhysicsContextId                   = 0;
    mGraphicsQueueType                  = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mPhysicsQueueType                   = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    mFrameActive                        = false;
    mNextPresentationReadbackRequestId  = 1;
    mNextPresentationReadbackFenceValue = 1;
    mNextPhysicsToGraphicsFenceValue    = 1;
    mNextGraphicsToPhysicsFenceValue    = 1;
    mPendingPresentationReadbackRequests.clear();
    mPendingPresentationReadbackCopies.clear();
    mCompletedPresentationReadbacks.clear();
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

bool GpuDeviceImpl::waitForGraphicsOnPhysics()
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

bool GpuDeviceImpl::supportsNativePhysicsFloatAtomics() const
{
    return mSupportsNativePhysicsFloatAtomics;
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

    Diligent::ShaderCreateInfo shaderCreateInfo = createInfo;
    Diligent::ShaderMacroHelper shaderMacros;
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

        const bool isPhysicsShader = shaderCreateInfo.FilePath != nullptr &&
                                     std::strstr(shaderCreateInfo.FilePath, "physics/") != nullptr;
        if (isPhysicsShader)
        {
            shaderMacros += shaderCreateInfo.Macros;
            shaderMacros.Add("CRESSIM_NATIVE_FLOAT_BUFFER_ATOMICS",
                             mSupportsNativePhysicsFloatAtomics);
            shaderCreateInfo.Macros = shaderMacros;
            if (mSupportsNativePhysicsFloatAtomics &&
                mDesc.vulkanShaderCompilerMode == VulkanShaderCompilerMode::Auto)
            {
                shaderCreateInfo.ShaderCompiler = Diligent::SHADER_COMPILER_DXC;
            }
        }
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

    const PhysicsFloatAtomicMode floatAtomicMode = mDesc.physicsFloatAtomicMode;
    const VulkanShaderCompilerMode shaderCompilerMode = mDesc.vulkanShaderCompilerMode;
    const std::string bundledDxCompilerPath = findBundledVulkanDXCompilerPath();
    const bool dxcConfiguredForVulkan = !bundledDxCompilerPath.empty();
    const bool shouldQueryFloatAtomicSupport = floatAtomicMode != PhysicsFloatAtomicMode::ForceCAS;
    const VulkanFloatAtomicAddSupport floatAtomicSupport =
        shouldQueryFloatAtomicSupport
            ? queryVulkanFloatAtomicAddSupport(Diligent::DEFAULT_ADAPTER_ID)
            : VulkanFloatAtomicAddSupport{};
    const bool canCompileNativePhysicsShaders =
        dxcConfiguredForVulkan && shaderCompilerMode != VulkanShaderCompilerMode::ForceDefault;

    if (shaderCompilerMode == VulkanShaderCompilerMode::ForceDXC && !dxcConfiguredForVulkan)
    {
        CRESSIM_LOG_ERROR(
            "Vulkan shader compiler mode is set to force DXC, but no bundled libdxcompiler.so "
            "was found next to the executable.");
        return false;
    }

    if (floatAtomicMode == PhysicsFloatAtomicMode::ForceNative)
    {
        if (!floatAtomicSupport.canUseNativePhysicsAtomics())
        {
            CRESSIM_LOG_ERROR(
                "physics float atomic mode is set to force native, but "
                "VK_EXT_shader_atomic_float with shaderBufferFloat32AtomicAdd is unavailable.");
            return false;
        }
        if (!canCompileNativePhysicsShaders)
        {
            CRESSIM_LOG_ERROR(
                "physics float atomic mode is set to force native, but Vulkan shaders are not "
                "configured to use bundled DXC.");
            return false;
        }
    }

    auto createDeviceContexts = [&](bool requestDedicatedPhysicsContext,
                                    bool enableNativeFloatAtomics)
    {
        mRenderDevice    = nullptr;
        mGraphicsContext = nullptr;
        mPhysicsContext  = nullptr;

        Diligent::EngineVkCreateInfo engineCreateInfo{};
        engineCreateInfo.EnableValidation =
            static_cast<Diligent::Bool>(mDesc.enableValidation ? 1 : 0);
        if (!bundledDxCompilerPath.empty())
        {
            engineCreateInfo.pDxCompilerPath = bundledDxCompilerPath.c_str();
        }
        std::array<const char *, 1> deviceExtensions{};
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shaderAtomicFloatFeatures{};
        if (enableNativeFloatAtomics)
        {
            deviceExtensions[0]               = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
            engineCreateInfo.DeviceExtensionCount = 1u;
            engineCreateInfo.ppDeviceExtensionNames = deviceExtensions.data();

            shaderAtomicFloatFeatures.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
            shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
            engineCreateInfo.pDeviceExtensionFeatures = &shaderAtomicFloatFeatures;
        }

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

    const auto createWithFallbackPlan = [&](bool enableNativeFloatAtomics)
    {
        if (createDeviceContexts(true, enableNativeFloatAtomics))
        {
            return true;
        }

        CRESSIM_LOG_WARNING(
            "failed to create dedicated physics context; falling back to shared context.");
        return createDeviceContexts(false, enableNativeFloatAtomics);
    };

    bool shouldAttemptNativeFloatAtomics = false;
    switch (floatAtomicMode)
    {
    case PhysicsFloatAtomicMode::ForceNative:
        shouldAttemptNativeFloatAtomics = true;
        break;
    case PhysicsFloatAtomicMode::Auto:
        shouldAttemptNativeFloatAtomics =
            floatAtomicSupport.canUseNativePhysicsAtomics() && canCompileNativePhysicsShaders;
        break;
    case PhysicsFloatAtomicMode::ForceCAS:
    default:
        break;
    }

    if (shouldAttemptNativeFloatAtomics)
    {
        CRESSIM_LOG_INFO("Attempting Vulkan device creation with native physics float atomics enabled.");
        if (createWithFallbackPlan(true))
        {
            mSupportsNativePhysicsFloatAtomics = true;
        }
        else if (floatAtomicMode == PhysicsFloatAtomicMode::ForceNative)
        {
            CRESSIM_LOG_ERROR(
                "physics float atomic mode is set to force native, but Vulkan device creation "
                "with native float atomics failed.");
            return false;
        }
    }

    if (mRenderDevice == nullptr)
    {
        if (floatAtomicMode == PhysicsFloatAtomicMode::Auto &&
            floatAtomicSupport.canUseNativePhysicsAtomics())
        {
            if (!dxcConfiguredForVulkan)
            {
                CRESSIM_LOG_WARNING(
                    "native physics float atomics are supported by Vulkan, but no bundled "
                    "libdxcompiler.so was found; using CAS fallback.");
            }
            else if (shaderCompilerMode == VulkanShaderCompilerMode::ForceDefault)
            {
                CRESSIM_LOG_WARNING(
                    "native physics float atomics are supported by Vulkan, but Vulkan shader "
                    "compiler mode forces the default compiler; using CAS fallback.");
            }
            else if (shouldAttemptNativeFloatAtomics)
            {
                CRESSIM_LOG_WARNING(
                    "native physics float atomics were attempted, but Vulkan device creation "
                    "failed; using CAS fallback.");
            }
        }

        if (!createWithFallbackPlan(false))
        {
            return false;
        }
    }

    CRESSIM_LOG_INFO("Physics float atomics: native=",
                     mSupportsNativePhysicsFloatAtomics ? "enabled" : "disabled",
                     ", mode=", physicsFloatAtomicModeToString(floatAtomicMode),
                     ", extensionSupport=",
                     shouldQueryFloatAtomicSupport
                         ? (floatAtomicSupport.extensionSupported ? "yes" : "no")
                         : "skipped",
                     ", shaderBufferFloat32AtomicAdd=",
                     shouldQueryFloatAtomicSupport
                         ? (floatAtomicSupport.bufferFloat32AtomicAdd ? "yes" : "no")
                         : "skipped",
                     ", shaderCompilerMode=", shaderCompilerModeToString(shaderCompilerMode),
                     ", dxcConfigured=", dxcConfiguredForVulkan ? "yes" : "no", ".");

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
