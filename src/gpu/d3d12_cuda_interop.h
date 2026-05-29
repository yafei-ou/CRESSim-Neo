#ifndef CRESSIM_NEO_GPU_D3D12_CUDA_INTEROP_H
#define CRESSIM_NEO_GPU_D3D12_CUDA_INTEROP_H

#include "gpu/cuda_interop_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#if defined(_WIN32)
#include "DiligentEngine/DiligentCore/Platforms/Win32/interface/WinHPostface.h"
#include "DiligentEngine/DiligentCore/Platforms/Win32/interface/WinHPreface.h"
#include <d3d12.h>
#include <wrl/client.h>
#endif

#include <cstdint>

namespace cressim::neo::gpu::d3d12interop
{

struct SharedBufferState
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> renderDevice;
#if defined(_WIN32)
    Microsoft::WRL::ComPtr<ID3D12Resource> d3d12Resource;
#endif
    bool ownsNativeAllocation = false;
};

struct TimelineSemaphoreState
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> renderDevice;
    Diligent::RefCntAutoPtr<Diligent::IFence> fence;
};

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
bool exportBufferHandle(const SharedBufferState &state, interop::NativeHandle &outHandle) noexcept;

bool createExportableTimelineSemaphore(Diligent::IRenderDevice *renderDevice, const char *name,
                                       TimelineSemaphoreState &state);
void resetExportableTimelineSemaphore(TimelineSemaphoreState &state) noexcept;
bool exportSemaphoreHandle(const TimelineSemaphoreState &state,
                           interop::NativeHandle &outHandle) noexcept;

} // namespace cressim::neo::gpu::d3d12interop

#endif // CRESSIM_NEO_GPU_D3D12_CUDA_INTEROP_H
