#include "gpu/shared_export_buffer.h"

#include "common/logger.h"
#include "gpu/gpu_buffer_utils.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/include/VulkanUtilities/VulkanHeaders.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/RenderDeviceVk.h"

#include <algorithm>

namespace cressim::neo::gpu
{

namespace
{

bool bindFlagsNeedUnorderedAccess(const Diligent::BIND_FLAGS bindFlags)
{
    return (bindFlags & Diligent::BIND_UNORDERED_ACCESS) != 0;
}

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

bool canUseVulkanExportableStructuredBuffer(const Diligent::IRenderDevice *renderDevice,
                                            const Diligent::USAGE usage,
                                            const Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                            const Diligent::Uint64 immediateContextMask,
                                            const std::uint32_t queueFamilyIndexCount)
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    (void)renderDevice;
    (void)usage;
    (void)cpuAccess;
    (void)immediateContextMask;
    (void)queueFamilyIndexCount;
    return false;
#else
    return renderDevice != nullptr &&
           renderDevice->GetDeviceInfo().Type == Diligent::RENDER_DEVICE_TYPE_VULKAN &&
           usage == Diligent::USAGE_DEFAULT && cpuAccess == Diligent::CPU_ACCESS_NONE &&
           (Diligent::PlatformMisc::CountOneBits(immediateContextMask) >= 1u) &&
           queueFamilyIndexCount <= 2u;
#endif
}

} // namespace

SharedExportBuffer::~SharedExportBuffer()
{
    reset();
}

bool SharedExportBuffer::ensureStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredElementCount, const std::uint32_t minimumCapacity,
    const Diligent::BIND_FLAGS bindFlags, const Diligent::USAGE usage,
    const Diligent::CPU_ACCESS_FLAGS cpuAccess, const Diligent::Uint64 immediateContextMask,
    const std::uint32_t *queueFamilyIndices, const std::uint32_t queueFamilyIndexCount)
{
    if (renderDevice == nullptr || elementStride == 0u)
    {
        return false;
    }

    const std::uint32_t requiredCapacity =
        std::max(requiredElementCount, std::max(minimumCapacity, 1u));
    if (mBuffer != nullptr && mCapacity >= requiredCapacity && mElementStride == elementStride)
    {
        return true;
    }

    return recreateStructuredBuffer(renderDevice, name, elementStride, requiredCapacity, bindFlags,
                                    usage, cpuAccess, immediateContextMask, queueFamilyIndices,
                                    queueFamilyIndexCount);
}

void SharedExportBuffer::reset()
{
    mBuffer = nullptr;
    resetNativeVulkanBuffer();
    mCapacity               = 0u;
    mElementStride          = 0u;
    mExportable             = false;
    mOwnsNativeVulkanBuffer = false;
    mRenderDevice           = nullptr;
}

bool SharedExportBuffer::exportOpaqueFd(int &outFd) const noexcept
{
    outFd = -1;

#if defined(__linux__)
    if (!mOwnsNativeVulkanBuffer || mVkMemory == nullptr || mRenderDevice == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{mRenderDevice,
                                                                      Diligent::IID_RenderDeviceVk};
    if (renderDeviceVk == nullptr)
    {
        return false;
    }

    const VkDevice vkDevice = renderDeviceVk->GetVkDevice();
    if (vkDevice == VK_NULL_HANDLE)
    {
        return false;
    }

    const auto getMemoryFd =
        reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(vkDevice, "vkGetMemoryFdKHR"));
    if (getMemoryFd == nullptr)
    {
        return false;
    }

    VkMemoryGetFdInfoKHR fdInfo{};
    fdInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory     = mVkMemory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int exportedFd = -1;
    if (getMemoryFd(vkDevice, &fdInfo, &exportedFd) != VK_SUCCESS || exportedFd < 0)
    {
        return false;
    }

    outFd = exportedFd;
    return true;
#else
    return false;
#endif
}

bool SharedExportBuffer::recreateStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredCapacity, const Diligent::BIND_FLAGS bindFlags,
    const Diligent::USAGE usage, const Diligent::CPU_ACCESS_FLAGS cpuAccess,
    const Diligent::Uint64 immediateContextMask, const std::uint32_t *queueFamilyIndices,
    const std::uint32_t queueFamilyIndexCount)
{
    reset();

    if (canUseVulkanExportableStructuredBuffer(renderDevice, usage, cpuAccess, immediateContextMask,
                                               queueFamilyIndexCount))
    {
        if (createVulkanExportableStructuredBuffer(
                renderDevice, name, elementStride, requiredCapacity, bindFlags, usage, cpuAccess,
                immediateContextMask, queueFamilyIndices, queueFamilyIndexCount))
        {
            return true;
        }

        CRESSIM_LOG_WARNING("Falling back to a non-exportable Diligent buffer for '", name,
                            "' after Vulkan shared-buffer allocation failed.");
    }

    return createGenericStructuredBuffer(renderDevice, name, elementStride, requiredCapacity,
                                         bindFlags, usage, cpuAccess, immediateContextMask);
}

bool SharedExportBuffer::createGenericStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredCapacity, const Diligent::BIND_FLAGS bindFlags,
    const Diligent::USAGE usage, const Diligent::CPU_ACCESS_FLAGS cpuAccess,
    const Diligent::Uint64 immediateContextMask)
{
    Diligent::RefCntAutoPtr<Diligent::IBuffer> buffer;
    std::uint32_t capacity = 0u;
    if (!detail::ensureStructuredBufferCapacity(renderDevice, name, elementStride, requiredCapacity,
                                                requiredCapacity, bindFlags, usage, cpuAccess,
                                                immediateContextMask, buffer, capacity) ||
        buffer == nullptr)
    {
        return false;
    }

    mBuffer                 = std::move(buffer);
    mCapacity               = capacity;
    mElementStride          = elementStride;
    mRenderDevice           = renderDevice;
    mExportable             = false;
    mOwnsNativeVulkanBuffer = false;
    return true;
}

bool SharedExportBuffer::createVulkanExportableStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredCapacity, const Diligent::BIND_FLAGS bindFlags,
    const Diligent::USAGE usage, const Diligent::CPU_ACCESS_FLAGS cpuAccess,
    const Diligent::Uint64 immediateContextMask, const std::uint32_t *queueFamilyIndices,
    const std::uint32_t queueFamilyIndexCount)
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{renderDevice,
                                                                      Diligent::IID_RenderDeviceVk};
    if (renderDeviceVk == nullptr)
    {
        return false;
    }

    const VkDevice vkDevice                 = renderDeviceVk->GetVkDevice();
    const VkPhysicalDevice vkPhysicalDevice = renderDeviceVk->GetVkPhysicalDevice();
    if (vkDevice == VK_NULL_HANDLE || vkPhysicalDevice == VK_NULL_HANDLE)
    {
        return false;
    }

#if defined(__linux__)
    constexpr VkExternalMemoryHandleTypeFlagBits kHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#elif defined(_WIN32)
    constexpr VkExternalMemoryHandleTypeFlagBits kHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    return false;
#endif

    VkExternalMemoryBufferCreateInfo externalBufferInfo{};
    externalBufferInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalBufferInfo.handleTypes = kHandleType;

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
    exportMemoryInfo.handleTypes = kHandleType;

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

    mBuffer                 = std::move(buffer);
    mCapacity               = requiredCapacity;
    mElementStride          = elementStride;
    mExportable             = true;
    mOwnsNativeVulkanBuffer = true;
    mVkBuffer               = vkBuffer;
    mVkMemory               = vkMemory;
    mRenderDevice           = renderDevice;
    CRESSIM_LOG_INFO("Created exportable Vulkan shared buffer '", name,
                     "' with capacity=", requiredCapacity, ", stride=", elementStride,
                     ", queueFamilies=", queueFamilyIndexCount > 0u ? queueFamilyIndexCount : 1u,
                     ".");
    return true;
}

void SharedExportBuffer::resetNativeVulkanBuffer() noexcept
{
    if (!mOwnsNativeVulkanBuffer || mRenderDevice == nullptr)
    {
        mVkBuffer               = nullptr;
        mVkMemory               = nullptr;
        mOwnsNativeVulkanBuffer = false;
        return;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{mRenderDevice,
                                                                      Diligent::IID_RenderDeviceVk};
    if (renderDeviceVk != nullptr)
    {
        mRenderDevice->IdleGPU();

        const VkDevice vkDevice = renderDeviceVk->GetVkDevice();
        if (vkDevice != VK_NULL_HANDLE)
        {
            if (mVkBuffer != nullptr)
            {
                vkDestroyBuffer(vkDevice, mVkBuffer, nullptr);
            }
            if (mVkMemory != nullptr)
            {
                vkFreeMemory(vkDevice, mVkMemory, nullptr);
            }
        }
    }

    mVkBuffer               = nullptr;
    mVkMemory               = nullptr;
    mOwnsNativeVulkanBuffer = false;
}

} // namespace cressim::neo::gpu
