#ifndef CRESSIM_NEO_GPU_VULKAN_CUDA_INTEROP_H
#define CRESSIM_NEO_GPU_VULKAN_CUDA_INTEROP_H

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#if defined(_WIN32) && !defined(VK_USE_PLATFORM_WIN32_KHR)
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif

#if DILIGENT_USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/RenderDeviceVk.h"

#include <cstdint>

namespace cressim::neo::gpu::vkinterop
{

struct NativeHandle
{
#if defined(_WIN32)
    void *win32Handle = nullptr;
#else
    int fd = -1;
#endif
};

struct SharedBufferState
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> renderDevice;
    VkBuffer vkBuffer         = VK_NULL_HANDLE;
    VkDeviceMemory vkMemory   = VK_NULL_HANDLE;
    bool ownsNativeAllocation = false;
};

struct TimelineSemaphoreState
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> renderDevice;
    Diligent::RefCntAutoPtr<Diligent::IFence> fence;
    VkSemaphore vkSemaphore = VK_NULL_HANDLE;
};

bool isValid(const NativeHandle &handle) noexcept;
void releaseOwnership(NativeHandle &handle) noexcept;
bool closeNativeHandle(NativeHandle &handle) noexcept;

bool canUseExportableStructuredBuffer(const Diligent::IRenderDevice *renderDevice,
                                      Diligent::USAGE usage, Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                      Diligent::Uint64 immediateContextMask,
                                      std::uint32_t queueFamilyIndexCount);

bool createExportableStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                                      std::uint32_t elementStride, std::uint32_t requiredCapacity,
                                      Diligent::BIND_FLAGS bindFlags, Diligent::USAGE usage,
                                      Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                      Diligent::Uint64 immediateContextMask,
                                      const std::uint32_t *queueFamilyIndices,
                                      std::uint32_t queueFamilyIndexCount, SharedBufferState &state,
                                      Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer);
void resetExportableStructuredBuffer(SharedBufferState &state) noexcept;
bool exportBufferHandle(const SharedBufferState &state, NativeHandle &outHandle) noexcept;

bool createExportableTimelineSemaphore(Diligent::IRenderDevice *renderDevice, const char *name,
                                       TimelineSemaphoreState &state);
void resetExportableTimelineSemaphore(TimelineSemaphoreState &state) noexcept;
bool exportSemaphoreHandle(const TimelineSemaphoreState &state, NativeHandle &outHandle) noexcept;

} // namespace cressim::neo::gpu::vkinterop

#endif // CRESSIM_NEO_GPU_VULKAN_CUDA_INTEROP_H
