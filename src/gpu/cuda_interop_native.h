#ifndef CRESSIM_NEO_GPU_CUDA_INTEROP_NATIVE_H
#define CRESSIM_NEO_GPU_CUDA_INTEROP_NATIVE_H

#include "gpu/cuda_interop_types.h"

#if defined(_WIN32)
#include "gpu/d3d12_cuda_interop.h"
#endif
#include "gpu/vulkan_cuda_interop.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

namespace cressim::neo::gpu::interop
{

struct SharedBufferState
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> renderDevice;
    Diligent::RENDER_DEVICE_TYPE deviceType = Diligent::RENDER_DEVICE_TYPE_UNDEFINED;
#if defined(_WIN32)
    d3d12interop::SharedBufferState d3d12State;
#endif
    VkBuffer vkBuffer         = VK_NULL_HANDLE;
    VkDeviceMemory vkMemory   = VK_NULL_HANDLE;
    bool ownsNativeAllocation = false;
};

struct TimelineSemaphoreState
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> renderDevice;
    Diligent::RefCntAutoPtr<Diligent::IFence> fence;
    Diligent::RENDER_DEVICE_TYPE deviceType = Diligent::RENDER_DEVICE_TYPE_UNDEFINED;
#if defined(_WIN32)
    d3d12interop::TimelineSemaphoreState d3d12State;
#endif
    VkSemaphore vkSemaphore = VK_NULL_HANDLE;
    bool initialized        = false;
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

} // namespace cressim::neo::gpu::interop

#endif // CRESSIM_NEO_GPU_CUDA_INTEROP_NATIVE_H
