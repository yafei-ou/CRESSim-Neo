#ifndef CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H
#define CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H

#include "gpu/cuda_interop_types.h"
#include "gpu/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"

#include <cstdint>
#include <memory>

namespace Diligent
{
struct IBuffer;
struct IRenderDevice;
} // namespace Diligent

namespace cressim::neo::gpu
{

class CudaSharedBuffer;

class CRESSIM_NEO_GPU_API SharedExportBuffer
{
public:
    SharedExportBuffer();
    ~SharedExportBuffer();

    SharedExportBuffer(const SharedExportBuffer &)            = delete;
    SharedExportBuffer &operator=(const SharedExportBuffer &) = delete;

    bool ensureStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                                std::uint32_t elementStride, std::uint32_t requiredElementCount,
                                std::uint32_t minimumCapacity, Diligent::BIND_FLAGS bindFlags,
                                Diligent::USAGE usage, Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                Diligent::Uint64 immediateContextMask,
                                const std::uint32_t *queueFamilyIndices = nullptr,
                                std::uint32_t queueFamilyIndexCount     = 0u);

    void reset();

    Diligent::IBuffer *buffer() const noexcept;
    const Diligent::RefCntAutoPtr<Diligent::IBuffer> &bufferRef() const noexcept;
    std::uint32_t capacity() const noexcept;
    std::uint32_t elementStride() const noexcept;
    std::uint64_t sizeBytes() const noexcept;
    bool isExportable() const noexcept;
    bool usesNativeSharedAllocation() const noexcept;

    Diligent::RENDER_DEVICE_TYPE nativeRenderDeviceType() const noexcept;

private:
    friend class CudaSharedBuffer;

    bool exportNativeHandle(interop::NativeHandle &outHandle) const noexcept;

    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H
