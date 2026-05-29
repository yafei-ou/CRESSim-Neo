#include "gpu/d3d12_cuda_interop.h"

#include "common/logger.h"
#include "gpu/cuda_interop_native.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineD3D12/interface/BufferD3D12.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineD3D12/interface/FenceD3D12.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineD3D12/interface/RenderDeviceD3D12.h"

namespace cressim::neo::gpu::d3d12interop
{

namespace
{

#if defined(_WIN32)
bool getRenderDeviceD3D12(const Diligent::RefCntAutoPtr<Diligent::IRenderDevice> &renderDevice,
                          Diligent::RefCntAutoPtr<Diligent::IRenderDeviceD3D12> &outRenderDevice)
{
    outRenderDevice = {renderDevice, Diligent::IID_RenderDeviceD3D12};
    return outRenderDevice != nullptr;
}

bool getRenderDeviceD3D12(Diligent::IRenderDevice *renderDevice,
                          Diligent::RefCntAutoPtr<Diligent::IRenderDeviceD3D12> &outRenderDevice)
{
    outRenderDevice = {renderDevice, Diligent::IID_RenderDeviceD3D12};
    return outRenderDevice != nullptr;
}

bool createD3D12SharedHandle(ID3D12Device *device, ID3D12DeviceChild *object,
                             interop::NativeHandle &outHandle) noexcept
{
    interop::releaseOwnership(outHandle);
    if (device == nullptr || object == nullptr)
    {
        return false;
    }

    HANDLE handle = nullptr;
    if (FAILED(device->CreateSharedHandle(object, nullptr, GENERIC_ALL, nullptr, &handle)) ||
        handle == nullptr)
    {
        return false;
    }

    outHandle.win32Handle = handle;
    return true;
}
#endif

} // namespace

bool canUseExportableStructuredBuffer(const Diligent::IRenderDevice *renderDevice,
                                      const Diligent::USAGE usage,
                                      const Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                      const Diligent::Uint64 immediateContextMask,
                                      const std::uint32_t queueFamilyIndexCount)
{
    (void)queueFamilyIndexCount;
    return renderDevice != nullptr &&
           renderDevice->GetDeviceInfo().Type == Diligent::RENDER_DEVICE_TYPE_D3D12 &&
           usage == Diligent::USAGE_DEFAULT && cpuAccess == Diligent::CPU_ACCESS_NONE &&
           Diligent::PlatformMisc::CountOneBits(immediateContextMask) >= 1u;
}

bool createExportableStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredCapacity, const Diligent::BIND_FLAGS bindFlags,
    const Diligent::USAGE usage, const Diligent::CPU_ACCESS_FLAGS cpuAccess,
    const Diligent::Uint64 immediateContextMask, const std::uint32_t *queueFamilyIndices,
    const std::uint32_t queueFamilyIndexCount, SharedBufferState &state,
    Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
#if !defined(_WIN32)
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
    resetExportableStructuredBuffer(state);
    outBuffer = nullptr;
    (void)queueFamilyIndices;
    (void)queueFamilyIndexCount;

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceD3D12> renderDeviceD3D12;
    if (!getRenderDeviceD3D12(renderDevice, renderDeviceD3D12))
    {
        return false;
    }

    ID3D12Device *device = renderDeviceD3D12->GetD3D12Device();
    if (device == nullptr)
    {
        return false;
    }

    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_NONE;
    if ((bindFlags & Diligent::BIND_UNORDERED_ACCESS) != 0)
    {
        resourceFlags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    const std::uint64_t sizeBytes = static_cast<std::uint64_t>(requiredCapacity) * elementStride;

    D3D12_HEAP_PROPERTIES heapProperties{};
    heapProperties.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    heapProperties.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProperties.CreationNodeMask     = 1u;
    heapProperties.VisibleNodeMask      = 1u;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment          = 0u;
    resourceDesc.Width              = sizeBytes;
    resourceDesc.Height             = 1u;
    resourceDesc.DepthOrArraySize   = 1u;
    resourceDesc.MipLevels          = 1u;
    resourceDesc.Format             = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count   = 1u;
    resourceDesc.SampleDesc.Quality = 0u;
    resourceDesc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags              = resourceFlags;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    const HRESULT createResourceResult = device->CreateCommittedResource(
        &heapProperties, D3D12_HEAP_FLAG_SHARED, &resourceDesc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(createResourceResult) || resource == nullptr)
    {
        CRESSIM_LOG_WARNING("CreateCommittedResource failed for exportable D3D12 shared buffer '",
                            name, "'.");
        return false;
    }

    Diligent::BufferDesc bufferDesc{};
    bufferDesc.Name                 = name;
    bufferDesc.Size                 = sizeBytes;
    bufferDesc.BindFlags            = bindFlags;
    bufferDesc.Usage                = usage;
    bufferDesc.CPUAccessFlags       = cpuAccess;
    bufferDesc.ImmediateContextMask = immediateContextMask;
    bufferDesc.Mode                 = Diligent::BUFFER_MODE_STRUCTURED;
    bufferDesc.ElementByteStride    = elementStride;

    renderDeviceD3D12->CreateBufferFromD3DResource(resource.Get(), bufferDesc,
                                                   Diligent::RESOURCE_STATE_COMMON, &outBuffer);
    if (outBuffer == nullptr)
    {
        CRESSIM_LOG_WARNING(
            "CreateBufferFromD3DResource failed for exportable D3D12 shared buffer '", name, "'.");
        return false;
    }

    state.renderDevice         = renderDevice;
    state.d3d12Resource        = resource;
    state.ownsNativeAllocation = true;
    CRESSIM_LOG_INFO("Created exportable D3D12 shared buffer '", name,
                     "' with capacity=", requiredCapacity, ", stride=", elementStride, ".");
    return true;
#endif
}

void resetExportableStructuredBuffer(SharedBufferState &state) noexcept
{
#if defined(_WIN32)
    if (state.renderDevice != nullptr)
    {
        state.renderDevice->IdleGPU();
    }
    state.d3d12Resource.Reset();
#endif
    state.renderDevice         = nullptr;
    state.ownsNativeAllocation = false;
}

bool exportBufferHandle(const SharedBufferState &state, interop::NativeHandle &outHandle) noexcept
{
#if !defined(_WIN32)
    (void)state;
    interop::releaseOwnership(outHandle);
    return false;
#else
    interop::releaseOwnership(outHandle);

    if (!state.ownsNativeAllocation || state.renderDevice == nullptr ||
        state.d3d12Resource == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceD3D12> renderDeviceD3D12;
    if (!getRenderDeviceD3D12(state.renderDevice, renderDeviceD3D12))
    {
        return false;
    }

    return createD3D12SharedHandle(renderDeviceD3D12->GetD3D12Device(), state.d3d12Resource.Get(),
                                   outHandle);
#endif
}

bool createExportableTimelineSemaphore(Diligent::IRenderDevice *renderDevice, const char *name,
                                       TimelineSemaphoreState &state)
{
#if !defined(_WIN32)
    (void)renderDevice;
    (void)name;
    (void)state;
    return false;
#else
    resetExportableTimelineSemaphore(state);

    Diligent::FenceDesc fenceDesc{};
    fenceDesc.Name = name;
    fenceDesc.Type = Diligent::FENCE_TYPE_GENERAL;

    Diligent::RefCntAutoPtr<Diligent::IFence> fence;
    renderDevice->CreateFence(fenceDesc, &fence);
    if (fence == nullptr)
    {
        return false;
    }

    state.renderDevice = renderDevice;
    state.fence        = std::move(fence);
    return true;
#endif
}

void resetExportableTimelineSemaphore(TimelineSemaphoreState &state) noexcept
{
    state.fence        = nullptr;
    state.renderDevice = nullptr;
}

bool exportSemaphoreHandle(const TimelineSemaphoreState &state,
                           interop::NativeHandle &outHandle) noexcept
{
#if !defined(_WIN32)
    (void)state;
    interop::releaseOwnership(outHandle);
    return false;
#else
    interop::releaseOwnership(outHandle);

    if (state.renderDevice == nullptr || state.fence == nullptr)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceD3D12> renderDeviceD3D12;
    Diligent::RefCntAutoPtr<Diligent::IFenceD3D12> fenceD3D12{state.fence,
                                                              Diligent::IID_FenceD3D12};
    if (!getRenderDeviceD3D12(state.renderDevice, renderDeviceD3D12) || fenceD3D12 == nullptr)
    {
        return false;
    }

    return createD3D12SharedHandle(renderDeviceD3D12->GetD3D12Device(), fenceD3D12->GetD3D12Fence(),
                                   outHandle);
#endif
}

} // namespace cressim::neo::gpu::d3d12interop
