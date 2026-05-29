#ifndef CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H
#define CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H

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

namespace vkinterop
{
struct NativeHandle;
}

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

    Diligent::IBuffer *buffer() const noexcept
    {
        return mBuffer.RawPtr();
    }

    const Diligent::RefCntAutoPtr<Diligent::IBuffer> &bufferRef() const noexcept
    {
        return mBuffer;
    }

    std::uint32_t capacity() const noexcept
    {
        return mCapacity;
    }

    std::uint32_t elementStride() const noexcept
    {
        return mElementStride;
    }

    std::uint64_t sizeBytes() const noexcept
    {
        return static_cast<std::uint64_t>(mCapacity) * mElementStride;
    }

    bool isExportable() const noexcept
    {
        return mExportable;
    }

    bool usesNativeSharedAllocation() const noexcept
    {
        return mUsesNativeSharedAllocation;
    }

private:
    friend class CudaSharedBuffer;

    bool exportNativeHandle(vkinterop::NativeHandle &outHandle) const noexcept;

    struct Impl;
    std::unique_ptr<Impl> mImpl;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBuffer;
    std::uint32_t mCapacity          = 0u;
    std::uint32_t mElementStride     = 0u;
    bool mExportable                 = false;
    bool mUsesNativeSharedAllocation = false;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H
