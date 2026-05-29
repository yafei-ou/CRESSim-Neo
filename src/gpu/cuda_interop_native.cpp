#include "gpu/cuda_interop_native.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

namespace cressim::neo::gpu::interop
{

namespace
{

vkinterop::NativeHandle toVulkanHandle(const NativeHandle &handle) noexcept
{
    vkinterop::NativeHandle vkHandle{};
#if defined(_WIN32)
    vkHandle.win32Handle = handle.win32Handle;
#else
    vkHandle.fd = handle.fd;
#endif
    return vkHandle;
}

NativeHandle fromVulkanHandle(const vkinterop::NativeHandle &handle) noexcept
{
    NativeHandle outHandle{};
#if defined(_WIN32)
    outHandle.win32Handle = handle.win32Handle;
#else
    outHandle.fd = handle.fd;
#endif
    return outHandle;
}

vkinterop::SharedBufferState toVulkanState(const SharedBufferState &state) noexcept
{
    vkinterop::SharedBufferState vkState{};
    vkState.renderDevice         = state.renderDevice;
    vkState.vkBuffer             = state.vkBuffer;
    vkState.vkMemory             = state.vkMemory;
    vkState.ownsNativeAllocation = state.ownsNativeAllocation;
    return vkState;
}

vkinterop::TimelineSemaphoreState toVulkanState(const TimelineSemaphoreState &state) noexcept
{
    vkinterop::TimelineSemaphoreState vkState{};
    vkState.renderDevice = state.renderDevice;
    vkState.fence        = state.fence;
    vkState.vkSemaphore  = state.vkSemaphore;
    return vkState;
}

void fromVulkanState(const vkinterop::SharedBufferState &vkState, SharedBufferState &state) noexcept
{
    state.renderDevice         = vkState.renderDevice;
    state.deviceType           = Diligent::RENDER_DEVICE_TYPE_VULKAN;
    state.vkBuffer             = vkState.vkBuffer;
    state.vkMemory             = vkState.vkMemory;
    state.ownsNativeAllocation = vkState.ownsNativeAllocation;
}

void fromVulkanState(const vkinterop::TimelineSemaphoreState &vkState,
                     TimelineSemaphoreState &state) noexcept
{
    state.renderDevice = vkState.renderDevice;
    state.fence        = vkState.fence;
    state.deviceType   = Diligent::RENDER_DEVICE_TYPE_VULKAN;
    state.vkSemaphore  = vkState.vkSemaphore;
    state.initialized  = vkState.fence != nullptr && vkState.vkSemaphore != VK_NULL_HANDLE;
}

} // namespace

bool isValid(const NativeHandle &handle) noexcept
{
    vkinterop::NativeHandle vkHandle = toVulkanHandle(handle);
    return vkinterop::isValid(vkHandle);
}

void releaseOwnership(NativeHandle &handle) noexcept
{
    vkinterop::NativeHandle vkHandle = toVulkanHandle(handle);
    vkinterop::releaseOwnership(vkHandle);
    handle = fromVulkanHandle(vkHandle);
}

bool closeNativeHandle(NativeHandle &handle) noexcept
{
    vkinterop::NativeHandle vkHandle = toVulkanHandle(handle);
    const bool closed                = vkinterop::closeNativeHandle(vkHandle);
    handle                           = fromVulkanHandle(vkHandle);
    return closed;
}

bool canUseExportableStructuredBuffer(const Diligent::IRenderDevice *renderDevice,
                                      const Diligent::USAGE usage,
                                      const Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                      const Diligent::Uint64 immediateContextMask,
                                      const std::uint32_t queueFamilyIndexCount)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    switch (renderDevice->GetDeviceInfo().Type)
    {
    case Diligent::RENDER_DEVICE_TYPE_VULKAN:
        return vkinterop::canUseExportableStructuredBuffer(
            renderDevice, usage, cpuAccess, immediateContextMask, queueFamilyIndexCount);
    case Diligent::RENDER_DEVICE_TYPE_D3D12:
        return d3d12interop::canUseExportableStructuredBuffer(
            renderDevice, usage, cpuAccess, immediateContextMask, queueFamilyIndexCount);
    default:
        return false;
    }
}

bool createExportableStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredCapacity, const Diligent::BIND_FLAGS bindFlags,
    const Diligent::USAGE usage, const Diligent::CPU_ACCESS_FLAGS cpuAccess,
    const Diligent::Uint64 immediateContextMask, const std::uint32_t *queueFamilyIndices,
    const std::uint32_t queueFamilyIndexCount, SharedBufferState &state,
    Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
    resetExportableStructuredBuffer(state);
    outBuffer = nullptr;

    if (renderDevice == nullptr)
    {
        return false;
    }

    if (renderDevice->GetDeviceInfo().Type == Diligent::RENDER_DEVICE_TYPE_VULKAN)
    {
        vkinterop::SharedBufferState vkState{};
        if (!vkinterop::createExportableStructuredBuffer(
                renderDevice, name, elementStride, requiredCapacity, bindFlags, usage, cpuAccess,
                immediateContextMask, queueFamilyIndices, queueFamilyIndexCount, vkState,
                outBuffer))
        {
            return false;
        }

        fromVulkanState(vkState, state);
        return true;
    }

    if (renderDevice->GetDeviceInfo().Type == Diligent::RENDER_DEVICE_TYPE_D3D12)
    {
        d3d12interop::SharedBufferState d3d12State{};
        if (!d3d12interop::createExportableStructuredBuffer(
                renderDevice, name, elementStride, requiredCapacity, bindFlags, usage, cpuAccess,
                immediateContextMask, queueFamilyIndices, queueFamilyIndexCount, d3d12State,
                outBuffer))
        {
            return false;
        }
        state.renderDevice         = renderDevice;
        state.deviceType           = Diligent::RENDER_DEVICE_TYPE_D3D12;
        state.d3d12State           = std::move(d3d12State);
        state.ownsNativeAllocation = state.d3d12State.ownsNativeAllocation;
        return true;
    }

    return false;
}

void resetExportableStructuredBuffer(SharedBufferState &state) noexcept
{
    switch (state.deviceType)
    {
    case Diligent::RENDER_DEVICE_TYPE_VULKAN:
    {
        vkinterop::SharedBufferState vkState = toVulkanState(state);
        vkinterop::resetExportableStructuredBuffer(vkState);
        break;
    }
    case Diligent::RENDER_DEVICE_TYPE_D3D12:
        d3d12interop::resetExportableStructuredBuffer(state.d3d12State);
        break;
    default:
        break;
    }

    state.renderDevice         = nullptr;
    state.deviceType           = Diligent::RENDER_DEVICE_TYPE_UNDEFINED;
    state.vkBuffer             = VK_NULL_HANDLE;
    state.vkMemory             = VK_NULL_HANDLE;
    state.ownsNativeAllocation = false;
}

bool exportBufferHandle(const SharedBufferState &state, NativeHandle &outHandle) noexcept
{
    releaseOwnership(outHandle);

    switch (state.deviceType)
    {
    case Diligent::RENDER_DEVICE_TYPE_VULKAN:
    {
        vkinterop::NativeHandle vkHandle{};
        if (!vkinterop::exportBufferHandle(toVulkanState(state), vkHandle))
        {
            return false;
        }
        outHandle = fromVulkanHandle(vkHandle);
        return true;
    }
    case Diligent::RENDER_DEVICE_TYPE_D3D12:
        return d3d12interop::exportBufferHandle(state.d3d12State, outHandle);
    default:
        return false;
    }
}

bool createExportableTimelineSemaphore(Diligent::IRenderDevice *renderDevice, const char *name,
                                       TimelineSemaphoreState &state)
{
    resetExportableTimelineSemaphore(state);

    if (renderDevice == nullptr)
    {
        return false;
    }

    if (renderDevice->GetDeviceInfo().Type == Diligent::RENDER_DEVICE_TYPE_VULKAN)
    {
        vkinterop::TimelineSemaphoreState vkState{};
        if (!vkinterop::createExportableTimelineSemaphore(renderDevice, name, vkState))
        {
            return false;
        }

        fromVulkanState(vkState, state);
        return true;
    }

    if (renderDevice->GetDeviceInfo().Type == Diligent::RENDER_DEVICE_TYPE_D3D12)
    {
        d3d12interop::TimelineSemaphoreState d3d12State{};
        if (!d3d12interop::createExportableTimelineSemaphore(renderDevice, name, d3d12State))
        {
            return false;
        }

        state.renderDevice = renderDevice;
        state.fence        = d3d12State.fence;
        state.deviceType   = Diligent::RENDER_DEVICE_TYPE_D3D12;
        state.d3d12State   = std::move(d3d12State);
        state.initialized  = true;
        return true;
    }

    return false;
}

void resetExportableTimelineSemaphore(TimelineSemaphoreState &state) noexcept
{
    state.fence        = nullptr;
    state.renderDevice = nullptr;
    state.deviceType   = Diligent::RENDER_DEVICE_TYPE_UNDEFINED;
#if defined(_WIN32)
    d3d12interop::resetExportableTimelineSemaphore(state.d3d12State);
#endif
    state.vkSemaphore = VK_NULL_HANDLE;
    state.initialized = false;
}

bool exportSemaphoreHandle(const TimelineSemaphoreState &state, NativeHandle &outHandle) noexcept
{
    releaseOwnership(outHandle);

    switch (state.deviceType)
    {
    case Diligent::RENDER_DEVICE_TYPE_VULKAN:
    {
        vkinterop::NativeHandle vkHandle{};
        if (!vkinterop::exportSemaphoreHandle(toVulkanState(state), vkHandle))
        {
            return false;
        }
        outHandle = fromVulkanHandle(vkHandle);
        return true;
    }
    case Diligent::RENDER_DEVICE_TYPE_D3D12:
        return d3d12interop::exportSemaphoreHandle(state.d3d12State, outHandle);
    default:
        return false;
    }
}

} // namespace cressim::neo::gpu::interop
