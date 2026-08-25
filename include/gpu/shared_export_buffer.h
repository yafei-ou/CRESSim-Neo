#ifndef CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H
#define CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H

#include "gpu/cuda_interop_types.h"
#include "gpu/export.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"

#include <cstdint>
#include <memory>

/// @file shared_export_buffer.h
/// @brief GPU buffer container supporting native OS handle export for external CUDA / compute interop.

namespace Diligent
{
struct IBuffer;
struct IRenderDevice;
} // namespace Diligent

namespace cressim::neo::gpu
{

class CudaSharedBuffer;

/// @brief Structured GPU buffer that can be shared across API boundaries via native OS memory handles.
class CRESSIM_NEO_GPU_API SharedExportBuffer
{
public:
    /// @brief Default constructor.
    SharedExportBuffer();
    /// @brief Destructor releasing the underlying buffer resource.
    ~SharedExportBuffer();

    SharedExportBuffer(const SharedExportBuffer &)            = delete;
    SharedExportBuffer &operator=(const SharedExportBuffer &) = delete;

    /// @brief Ensures a structured buffer of at least the specified element capacity exists, allocating or reallocating if needed.
    /// @param renderDevice Graphics device interface.
    /// @param name Debug identifier for the buffer.
    /// @param elementStride Stride of an individual element in bytes.
    /// @param requiredElementCount Minimum required element count.
    /// @param minimumCapacity Capacity floor for power-of-two growth.
    /// @param bindFlags Buffer binding flags (e.g. BIND_UNORDERED_ACCESS | BIND_SHADER_RESOURCE).
    /// @param usage Buffer memory usage (e.g. USAGE_DEFAULT).
    /// @param cpuAccess Allowed CPU access flags.
    /// @param immediateContextMask Bitmask of device contexts that can access this buffer.
    /// @param queueFamilyIndices Optional array of Vulkan queue family indices for concurrent sharing.
    /// @param queueFamilyIndexCount Number of queue family indices.
    /// @return True on success.
    bool ensureStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                                std::uint32_t elementStride, std::uint32_t requiredElementCount,
                                std::uint32_t minimumCapacity, Diligent::BIND_FLAGS bindFlags,
                                Diligent::USAGE usage, Diligent::CPU_ACCESS_FLAGS cpuAccess,
                                Diligent::Uint64 immediateContextMask,
                                const std::uint32_t *queueFamilyIndices = nullptr,
                                std::uint32_t queueFamilyIndexCount     = 0u);

    /// @brief Releases the underlying GPU buffer.
    void reset();

    /// @brief Retrieves the raw Diligent buffer pointer.
    /// @return Pointer to Diligent::IBuffer.
    Diligent::IBuffer *buffer() const noexcept;
    /// @brief Retrieves the reference-counted Diligent buffer smart pointer.
    /// @return Const reference to RefCntAutoPtr<Diligent::IBuffer>.
    const Diligent::RefCntAutoPtr<Diligent::IBuffer> &bufferRef() const noexcept;
    /// @brief Retrieves the allocated element capacity.
    /// @return Element capacity.
    std::uint32_t capacity() const noexcept;
    /// @brief Retrieves the element stride in bytes.
    /// @return Byte stride per element.
    std::uint32_t elementStride() const noexcept;
    /// @brief Retrieves the total buffer size in bytes.
    /// @return Total byte size.
    std::uint64_t sizeBytes() const noexcept;
    /// @brief Checks if the buffer was created with OS export flags enabled.
    /// @return True if exportable.
    bool isExportable() const noexcept;
    /// @brief Checks if native shared allocation memory was used.
    /// @return True if native shared allocation active.
    bool usesNativeSharedAllocation() const noexcept;

    /// @brief Retrieves the native rendering device type hosting this buffer.
    /// @return Diligent::RENDER_DEVICE_TYPE.
    Diligent::RENDER_DEVICE_TYPE nativeRenderDeviceType() const noexcept;

private:
    friend class CudaSharedBuffer;

    bool exportNativeHandle(interop::NativeHandle &outHandle) const noexcept;

    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_SHARED_EXPORT_BUFFER_H
