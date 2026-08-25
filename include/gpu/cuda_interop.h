#ifndef CRESSIM_NEO_GPU_CUDA_INTEROP_H
#define CRESSIM_NEO_GPU_CUDA_INTEROP_H

#include "gpu/export.h"

#include <cstdint>
#include <memory>

/// @file cuda_interop.h
/// @brief CUDA / Direct3D12 / Vulkan external memory interop and timeline semaphore synchronization
/// primitives.

namespace Diligent
{
struct IDeviceContext;
struct IFence;
struct IRenderDevice;
} // namespace Diligent

namespace cressim::neo::gpu
{

class SharedExportBuffer;

/// @brief Opaque handle representing a native CUDA execution stream (`cudaStream_t`).
using CudaStreamHandle = void *;

/// @brief Wrapper encapsulating a CUDA asynchronous stream and execution queue.
class CRESSIM_NEO_GPU_API CudaStream
{
public:
    /// @brief Default constructor.
    CudaStream();
    /// @brief Destructor releasing CUDA stream resources.
    ~CudaStream();

    CudaStream(const CudaStream &)            = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    /// @brief Creates and initializes the underlying CUDA stream.
    /// @return True on success, false on failure or if CUDA interop is unsupported.
    bool initialize();
    /// @brief Destroys and resets the active CUDA stream.
    void reset();

    /// @brief Checks if the CUDA stream has been successfully initialized.
    /// @return True if initialized.
    bool isInitialized() const noexcept;
    /// @brief Retrieves the raw native CUDA stream handle.
    /// @return Pointer to native `cudaStream_t` or nullptr.
    CudaStreamHandle handle() const noexcept;
    /// @brief Blocks the host CPU thread until all queued stream work completes.
    /// @return True on success.
    bool synchronize();
    /// @brief Queues an asynchronous device-to-host memory copy on this stream.
    /// @param dst Destination host memory buffer pointer.
    /// @param src Source CUDA device memory pointer.
    /// @param sizeBytes Number of bytes to transfer.
    /// @return True if copy successfully enqueued.
    bool copyDeviceToHostAsync(void *dst, const void *src, std::uint64_t sizeBytes);

    /// @brief Checks whether the current build configuration supports CUDA interop.
    /// @return True if CUDA interop features are compiled in.
    static bool supportsCudaInteropBuild() noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief External timeline semaphore bridging Vulkan/D3D12 GPU timeline fences with CUDA stream
/// synchronization.
class CRESSIM_NEO_GPU_API CudaExternalTimelineSemaphore
{
public:
    /// @brief Default constructor.
    CudaExternalTimelineSemaphore();
    /// @brief Destructor releasing fence and semaphore handles.
    ~CudaExternalTimelineSemaphore();

    CudaExternalTimelineSemaphore(const CudaExternalTimelineSemaphore &)            = delete;
    CudaExternalTimelineSemaphore &operator=(const CudaExternalTimelineSemaphore &) = delete;

    /// @brief Allocates an exportable graphics timeline fence on the render device.
    /// @param renderDevice Graphics device interface.
    /// @param name Debug identifier for the semaphore.
    /// @return True on success.
    bool initialize(Diligent::IRenderDevice *renderDevice, const char *name);
    /// @brief Releases the underlying fence and imported CUDA semaphore.
    void reset();

    /// @brief Checks if the timeline semaphore has been allocated on the graphics device.
    /// @return True if initialized.
    bool isInitialized() const noexcept;
    /// @brief Checks if the external semaphore has been imported into the CUDA driver.
    /// @return True if imported.
    bool isImportedIntoCuda() const noexcept;

    /// @brief Imports the exported native OS handle into the CUDA driver as an external semaphore.
    /// @return True on success.
    bool importIntoCuda();

    /// @brief Queues a graphics-side fence signal on the specified device context.
    /// @param context Device context executing the signal.
    /// @param value Monotonically increasing 64-bit value to signal.
    /// @return True on success.
    bool signalOnDeviceContext(Diligent::IDeviceContext *context, std::uint64_t value);
    /// @brief Queues a graphics-side fence wait on the specified device context.
    /// @param context Device context waiting on the fence.
    /// @param value 64-bit timeline value to wait for.
    /// @return True on success.
    bool waitOnDeviceContext(Diligent::IDeviceContext *context, std::uint64_t value);
    /// @brief Queues a CUDA-side external semaphore signal on the given CUDA stream.
    /// @param stream Target CUDA stream.
    /// @param value 64-bit timeline value to signal.
    /// @return True on success.
    bool signalOnCudaStream(CudaStreamHandle stream, std::uint64_t value);
    /// @brief Queues a CUDA-side external semaphore wait on the given CUDA stream.
    /// @param stream Target CUDA stream.
    /// @param value 64-bit timeline value to wait for.
    /// @return True on success.
    bool waitOnCudaStream(CudaStreamHandle stream, std::uint64_t value);

    /// @brief Gets the underlying Diligent timeline fence pointer.
    /// @return Pointer to Diligent::IFence.
    Diligent::IFence *fence() const noexcept;
    /// @brief Gets the imported CUDA external semaphore handle (`cudaExternalSemaphore_t`).
    /// @return Pointer to CUDA semaphore handle.
    void *cudaSemaphoreHandle() const noexcept;

    /// @brief Checks if CUDA interop build support is available.
    /// @return True if supported.
    static bool supportsCudaInteropBuild() noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief Imported CUDA device memory view wrapping an exported graphics buffer.
class CRESSIM_NEO_GPU_API CudaSharedBuffer
{
public:
    /// @brief Default constructor.
    CudaSharedBuffer();
    /// @brief Destructor releasing imported CUDA memory mappings.
    ~CudaSharedBuffer();

    CudaSharedBuffer(const CudaSharedBuffer &)            = delete;
    CudaSharedBuffer &operator=(const CudaSharedBuffer &) = delete;

    /// @brief Imports an exportable graphics buffer into CUDA device memory space.
    /// @param buffer Source exportable shared graphics buffer.
    /// @return True on success.
    bool importFromSharedExportBuffer(const SharedExportBuffer &buffer);
    /// @brief Unmaps and releases the imported CUDA memory.
    void reset();

    /// @brief Checks if the buffer is currently imported and mapped into CUDA.
    /// @return True if imported.
    bool isImported() const noexcept;
    /// @brief Retrieves the CUDA linear device memory pointer.
    /// @return Raw device pointer (`void *`) accessible by CUDA kernels.
    void *devicePointer() const noexcept;
    /// @brief Retrieves the size of the mapped device memory in bytes.
    /// @return Buffer size in bytes.
    std::uint64_t sizeBytes() const noexcept;
    /// @brief Retrieves the CUDA device ordinal index hosting the buffer.
    /// @return Zero-based CUDA device index.
    std::int32_t deviceOrdinal() const noexcept;

    /// @brief Checks if CUDA interop build support is available.
    /// @return True if supported.
    static bool supportsCudaInteropBuild() noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

/// @brief High-level synchronization bridge managing shared graphics/CUDA buffers, streams, and
/// timeline fences.
class CRESSIM_NEO_GPU_API CudaSharedBufferBridge
{
public:
    /// @brief Default constructor.
    CudaSharedBufferBridge();
    /// @brief Destructor releasing bridge resources.
    ~CudaSharedBufferBridge();

    CudaSharedBufferBridge(const CudaSharedBufferBridge &)            = delete;
    CudaSharedBufferBridge &operator=(const CudaSharedBufferBridge &) = delete;

    /// @brief Initializes synchronization timeline semaphores on the graphics device.
    /// @param renderDevice Graphics device interface.
    /// @param name Debug identifier.
    /// @return True on success.
    bool initialize(Diligent::IRenderDevice *renderDevice, const char *name);
    /// @brief Resets bridge state and releases imported buffers and streams.
    void reset();

    /// @brief Checks if the bridge is initialized.
    /// @return True if initialized.
    bool isInitialized() const noexcept;
    /// @brief Binds and maps an exportable shared graphics buffer into this bridge.
    /// @param buffer Source shared export buffer.
    /// @return True on success.
    bool bindSharedBuffer(const SharedExportBuffer &buffer);
    /// @brief Signals from CUDA and instructs the graphics device context to wait for CUDA work
    /// completion.
    /// @param context Device context to synchronize.
    /// @return True on success.
    bool synchronizeToDeviceContext(Diligent::IDeviceContext *context);
    /// @brief Signals from graphics and instructs the CUDA stream to wait for graphics work
    /// completion.
    /// @param context Device context executing the graphics work.
    /// @return True on success.
    bool synchronizeFromDeviceContext(Diligent::IDeviceContext *context);
    /// @brief Asynchronously copies device memory from the shared buffer to host memory via the
    /// internal CUDA stream.
    /// @param dst Destination host memory pointer.
    /// @param src Source CUDA device pointer.
    /// @param sizeBytes Transfer size in bytes.
    /// @return True on success.
    bool copyDeviceToHostAsync(void *dst, const void *src, std::uint64_t sizeBytes);
    /// @brief Synchronizes the internal CUDA stream until all pending CUDA commands complete.
    /// @return True on success.
    bool synchronizeStream();

    /// @brief Retrieves the internal CUDA stream handle.
    /// @return CUDA stream handle.
    CudaStreamHandle streamHandle() const noexcept;
    /// @brief Retrieves the CUDA device memory pointer for the bound buffer.
    /// @return Device pointer (`void *`).
    void *devicePointer() const noexcept;
    /// @brief Retrieves the bound buffer size in bytes.
    /// @return Size in bytes.
    std::uint64_t sizeBytes() const noexcept;
    /// @brief Retrieves the CUDA device ordinal hosting the buffer.
    /// @return CUDA device index.
    std::int32_t deviceOrdinal() const noexcept;

    /// @brief Checks if CUDA interop is supported in this build.
    /// @return True if supported.
    static bool supportsCudaInteropBuild() noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_CUDA_INTEROP_H
