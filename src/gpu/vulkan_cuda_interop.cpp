#include "gpu/vulkan_cuda_interop.h"

#include "common/logger.h"

#include "DiligentEngine/DiligentCore/Platforms/interface/PlatformMisc.hpp"

#if defined(_WIN32)
#include <Windows.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

namespace cressim::neo::gpu::vkinterop
{

namespace
{

VkBufferUsageFlags toVkBufferUsage(const Diligent::BIND_FLAGS bindFlags)
{
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if ((bindFlags & Diligent::BIND_SHADER_RESOURCE) != 0 ||
        (bindFlags & Diligent::BIND_UNORDERED_ACCESS) != 0)
    {
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    return usage;
}

bool bindFlagsNeedUnorderedAccess(const Diligent::BIND_FLAGS bindFlags)
{
    return (bindFlags & Diligent::BIND_UNORDERED_ACCESS) != 0;
}

bool findMemoryTypeIndex(const VkPhysicalDeviceMemoryProperties &memoryProperties,
                         const std::uint32_t typeBits, const VkMemoryPropertyFlags preferredFlags,
                         std::uint32_t &outMemoryTypeIndex)
{
    for (std::uint32_t index = 0u; index < memoryProperties.memoryTypeCount; ++index)
    {
        const bool typeAllowed = (typeBits & (1u << index)) != 0u;
        if (!typeAllowed)
        {
            continue;
        }

        const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((flags & preferredFlags) == preferredFlags)
        {
            outMemoryTypeIndex = index;
            return true;
        }
    }

    for (std::uint32_t index = 0u; index < memoryProperties.memoryTypeCount; ++index)
    {
        if ((typeBits & (1u << index)) != 0u)
        {
            outMemoryTypeIndex = index;
            return true;
        }
    }

    return false;
}

bool getRenderDeviceVk(const Diligent::RefCntAutoPtr<Diligent::IRenderDevice> &renderDevice,
                       Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> &outRenderDeviceVk)
{
    outRenderDeviceVk = {renderDevice, Diligent::IID_RenderDeviceVk};
    return outRenderDeviceVk != nullptr;
}

bool getRenderDeviceVk(Diligent::IRenderDevice *renderDevice,
                       Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> &outRenderDeviceVk)
{
    outRenderDeviceVk = {renderDevice, Diligent::IID_RenderDeviceVk};
    return outRenderDeviceVk != nullptr;
}

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemoryHandleType =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kExternalSemaphoreHandleType =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

} // namespace

bool isValid(const NativeHandle &handle) noexcept
{
#if defined(_WIN32)
    return handle.win32Handle != nullptr;
#else
    return handle.fd >= 0;
#endif
}

void releaseOwnership(NativeHandle &handle) noexcept
{
#if defined(_WIN32)
    handle.win32Handle = nullptr;
#else
    handle.fd = -1;
#endif
}

bool closeNativeHandle(NativeHandle &handle) noexcept
{
#if defined(_WIN32)
    if (handle.win32Handle == nullptr)
    {
        return true;
    }

    const HANDLE nativeHandle = static_cast<HANDLE>(handle.win32Handle);
    handle.win32Handle        = nullptr;
    return CloseHandle(nativeHandle) != 0;
#elif defined(__linux__)
    if (handle.fd < 0)
    {
        return true;
    }

    const int fd = handle.fd;
    handle.fd    = -1;
    return close(fd) == 0;
#else
    releaseOwnership(handle);
    return true;
#endif
}

bool canUseExportableStructuredBuffer(const Diligent::IRenderDevice *renderDevice,
                                      const Diligent::USAGE usage,
                                      const Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                      const Diligent::Uint64 immediateContextMask,
                                      const std::uint32_t queueFamilyIndexCount)
{
    return renderDevice != nullptr &&
           renderDevice->GetDeviceInfo().Type == Diligent::RENDER_DEVICE_TYPE_VULKAN &&
           usage == Diligent::USAGE_DEFAULT && cpuAccess == Diligent::CPU_ACCESS_NONE &&
           Diligent::PlatformMisc::CountOneBits(immediateContextMask) >= 1u &&
           queueFamilyIndexCount <= 2u;
}

bool createExportableStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredCapacity, const Diligent::BIND_FLAGS bindFlags,
    const Diligent::USAGE usage, const Diligent::CPU_ACCESS_FLAGS cpuAccess,
    const Diligent::Uint64 immediateContextMask, const std::uint32_t *queueFamilyIndices,
    const std::uint32_t queueFamilyIndexCount, SharedBufferState &state,
    Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
#if !defined(_WIN32) && !defined(__linux__)
    (void)renderDevice;
    (void)name;
    (void)elementStride;
    (void)requiredCapacity;
    (void)bindFlags;
    (void)usage;
    (void)cpuAccess;
    (void)immediateContextMask;
    (void)queueFamilyIndices;
    (void)queueFamilyIndexCount;
    (void)state;
    (void)outBuffer;
    return false;
#else
    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk;
    if (!getRenderDeviceVk(renderDevice, renderDeviceVk))
    {
        return false;
    }

    const VkDevice vkDevice                 = renderDeviceVk->GetVkDevice();
    const VkPhysicalDevice vkPhysicalDevice = renderDeviceVk->GetVkPhysicalDevice();
    if (vkDevice == VK_NULL_HANDLE || vkPhysicalDevice == VK_NULL_HANDLE)
    {
        return false;
    }

    VkExternalMemoryBufferCreateInfo externalBufferInfo{};
    externalBufferInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalBufferInfo.handleTypes = kExternalMemoryHandleType;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &externalBufferInfo;
    bufferInfo.size  = static_cast<VkDeviceSize>(requiredCapacity) * elementStride;
    bufferInfo.usage = toVkBufferUsage(bindFlags);
    if (bindFlagsNeedUnorderedAccess(bindFlags))
    {
        bufferInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }

    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (queueFamilyIndices != nullptr && queueFamilyIndexCount > 1u)
    {
        bufferInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = queueFamilyIndexCount;
        bufferInfo.pQueueFamilyIndices   = queueFamilyIndices;
    }

    VkBuffer vkBuffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &vkBuffer) != VK_SUCCESS ||
        vkBuffer == VK_NULL_HANDLE)
    {
        CRESSIM_LOG_WARNING("vkCreateBuffer failed for exportable shared buffer '", name, "'.");
        return false;
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(vkDevice, vkBuffer, &memoryRequirements);

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memoryProperties);

    std::uint32_t memoryTypeIndex = 0u;
    if (!findMemoryTypeIndex(memoryProperties, memoryRequirements.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memoryTypeIndex))
    {
        vkDestroyBuffer(vkDevice, vkBuffer, nullptr);
        return false;
    }

    VkExportMemoryAllocateInfo exportMemoryInfo{};
    exportMemoryInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportMemoryInfo.handleTypes = kExternalMemoryHandleType;

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = &exportMemoryInfo;
    allocateInfo.allocationSize  = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    if (vkAllocateMemory(vkDevice, &allocateInfo, nullptr, &vkMemory) != VK_SUCCESS ||
        vkMemory == VK_NULL_HANDLE)
    {
        vkDestroyBuffer(vkDevice, vkBuffer, nullptr);
        CRESSIM_LOG_WARNING("vkAllocateMemory failed for exportable shared buffer '", name, "'.");
        return false;
    }

    if (vkBindBufferMemory(vkDevice, vkBuffer, vkMemory, 0u) != VK_SUCCESS)
    {
        vkFreeMemory(vkDevice, vkMemory, nullptr);
        vkDestroyBuffer(vkDevice, vkBuffer, nullptr);
        CRESSIM_LOG_WARNING("vkBindBufferMemory failed for exportable shared buffer '", name, "'.");
        return false;
    }

    Diligent::BufferDesc bufferDesc{};
    bufferDesc.Name           = name;
    bufferDesc.Size           = static_cast<Diligent::Uint64>(requiredCapacity) * elementStride;
    bufferDesc.BindFlags      = bindFlags;
    bufferDesc.Usage          = usage;
    bufferDesc.CPUAccessFlags = cpuAccess;
    bufferDesc.ImmediateContextMask = immediateContextMask;
    bufferDesc.Mode                 = Diligent::BUFFER_MODE_STRUCTURED;
    bufferDesc.ElementByteStride    = elementStride;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> buffer;
    renderDeviceVk->CreateBufferFromVulkanResource(vkBuffer, bufferDesc,
                                                   Diligent::RESOURCE_STATE_UNKNOWN, &buffer);
    if (buffer == nullptr)
    {
        vkFreeMemory(vkDevice, vkMemory, nullptr);
        vkDestroyBuffer(vkDevice, vkBuffer, nullptr);
        CRESSIM_LOG_WARNING("CreateBufferFromVulkanResource failed for exportable shared buffer '",
                            name, "'.");
        return false;
    }

    state.renderDevice         = renderDevice;
    state.vkBuffer             = vkBuffer;
    state.vkMemory             = vkMemory;
    state.ownsNativeAllocation = true;
    outBuffer                  = std::move(buffer);
    CRESSIM_LOG_INFO("Created exportable Vulkan shared buffer '", name,
                     "' with capacity=", requiredCapacity, ", stride=", elementStride,
                     ", queueFamilies=", queueFamilyIndexCount > 0u ? queueFamilyIndexCount : 1u,
                     ".");
    return true;
#endif
}

void resetExportableStructuredBuffer(SharedBufferState &state) noexcept
{
    if (!state.ownsNativeAllocation || state.renderDevice == nullptr)
    {
        state.vkBuffer             = VK_NULL_HANDLE;
        state.vkMemory             = VK_NULL_HANDLE;
        state.ownsNativeAllocation = false;
        return;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk;
    if (getRenderDeviceVk(state.renderDevice, renderDeviceVk))
    {
        state.renderDevice->IdleGPU();

        const VkDevice vkDevice = renderDeviceVk->GetVkDevice();
        if (vkDevice != VK_NULL_HANDLE && state.vkMemory != VK_NULL_HANDLE)
        {
            vkFreeMemory(vkDevice, state.vkMemory, nullptr);
        }
    }

    state.vkBuffer             = VK_NULL_HANDLE;
    state.vkMemory             = VK_NULL_HANDLE;
    state.ownsNativeAllocation = false;
    state.renderDevice         = nullptr;
}

bool exportBufferHandle(const SharedBufferState &state, NativeHandle &outHandle) noexcept
{
    releaseOwnership(outHandle);

#if defined(_WIN32)
    if (!state.ownsNativeAllocation || state.vkMemory == VK_NULL_HANDLE ||
        state.renderDevice == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk;
    if (!getRenderDeviceVk(state.renderDevice, renderDeviceVk))
    {
        return false;
    }

    const VkDevice vkDevice    = renderDeviceVk->GetVkDevice();
    const auto getMemoryHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(vkDevice, "vkGetMemoryWin32HandleKHR"));
    if (vkDevice == VK_NULL_HANDLE || getMemoryHandle == nullptr)
    {
        return false;
    }

    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory     = state.vkMemory;
    handleInfo.handleType = kExternalMemoryHandleType;

    HANDLE win32Handle = nullptr;
    if (getMemoryHandle(vkDevice, &handleInfo, &win32Handle) != VK_SUCCESS ||
        win32Handle == nullptr)
    {
        return false;
    }

    outHandle.win32Handle = win32Handle;
    return true;
#elif defined(__linux__)
    if (!state.ownsNativeAllocation || state.vkMemory == VK_NULL_HANDLE ||
        state.renderDevice == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk;
    if (!getRenderDeviceVk(state.renderDevice, renderDeviceVk))
    {
        return false;
    }

    const VkDevice vkDevice = renderDeviceVk->GetVkDevice();
    const auto getMemoryFd =
        reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(vkDevice, "vkGetMemoryFdKHR"));
    if (vkDevice == VK_NULL_HANDLE || getMemoryFd == nullptr)
    {
        return false;
    }

    VkMemoryGetFdInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    handleInfo.memory     = state.vkMemory;
    handleInfo.handleType = kExternalMemoryHandleType;

    int fd = -1;
    if (getMemoryFd(vkDevice, &handleInfo, &fd) != VK_SUCCESS || fd < 0)
    {
        return false;
    }

    outHandle.fd = fd;
    return true;
#else
    (void)state;
    return false;
#endif
}

bool createExportableTimelineSemaphore(Diligent::IRenderDevice *renderDevice, const char *name,
                                       TimelineSemaphoreState &state)
{
#if !defined(_WIN32) && !defined(__linux__)
    (void)renderDevice;
    (void)name;
    (void)state;
    return false;
#else
    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk;
    if (!getRenderDeviceVk(renderDevice, renderDeviceVk))
    {
        return false;
    }

    const VkDevice vkDevice = renderDeviceVk->GetVkDevice();
    if (vkDevice == VK_NULL_HANDLE)
    {
        return false;
    }

    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue  = 0u;

    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = kExternalSemaphoreHandleType;
    exportInfo.pNext       = &timelineInfo;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &exportInfo;

    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS ||
        semaphore == VK_NULL_HANDLE)
    {
        return false;
    }

    Diligent::FenceDesc fenceDesc{};
    fenceDesc.Name = name;
    fenceDesc.Type = Diligent::FENCE_TYPE_GENERAL;

    Diligent::RefCntAutoPtr<Diligent::IFence> fence;
    renderDeviceVk->CreateFenceFromVulkanResource(semaphore, fenceDesc, &fence);
    if (fence == nullptr)
    {
        vkDestroySemaphore(vkDevice, semaphore, nullptr);
        return false;
    }

    state.renderDevice = renderDevice;
    state.fence        = std::move(fence);
    state.vkSemaphore  = semaphore;
    return true;
#endif
}

void resetExportableTimelineSemaphore(TimelineSemaphoreState &state) noexcept
{
    state.fence        = nullptr;
    state.vkSemaphore  = VK_NULL_HANDLE;
    state.renderDevice = nullptr;
}

bool exportSemaphoreHandle(const TimelineSemaphoreState &state, NativeHandle &outHandle) noexcept
{
    releaseOwnership(outHandle);

#if defined(_WIN32)
    if (state.renderDevice == nullptr || state.vkSemaphore == VK_NULL_HANDLE)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk;
    if (!getRenderDeviceVk(state.renderDevice, renderDeviceVk))
    {
        return false;
    }

    const VkDevice vkDevice       = renderDeviceVk->GetVkDevice();
    const auto getSemaphoreHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(vkDevice, "vkGetSemaphoreWin32HandleKHR"));
    if (vkDevice == VK_NULL_HANDLE || getSemaphoreHandle == nullptr)
    {
        return false;
    }

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore  = state.vkSemaphore;
    handleInfo.handleType = kExternalSemaphoreHandleType;

    HANDLE win32Handle = nullptr;
    if (getSemaphoreHandle(vkDevice, &handleInfo, &win32Handle) != VK_SUCCESS ||
        win32Handle == nullptr)
    {
        return false;
    }

    outHandle.win32Handle = win32Handle;
    return true;
#elif defined(__linux__)
    if (state.renderDevice == nullptr || state.vkSemaphore == VK_NULL_HANDLE)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk;
    if (!getRenderDeviceVk(state.renderDevice, renderDeviceVk))
    {
        return false;
    }

    const VkDevice vkDevice   = renderDeviceVk->GetVkDevice();
    const auto getSemaphoreFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(vkDevice, "vkGetSemaphoreFdKHR"));
    if (vkDevice == VK_NULL_HANDLE || getSemaphoreFd == nullptr)
    {
        return false;
    }

    VkSemaphoreGetFdInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    handleInfo.semaphore  = state.vkSemaphore;
    handleInfo.handleType = kExternalSemaphoreHandleType;

    int fd = -1;
    if (getSemaphoreFd(vkDevice, &handleInfo, &fd) != VK_SUCCESS || fd < 0)
    {
        return false;
    }

    outHandle.fd = fd;
    return true;
#else
    (void)state;
    return false;
#endif
}

} // namespace cressim::neo::gpu::vkinterop
