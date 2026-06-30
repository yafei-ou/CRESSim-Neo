#ifndef CRESSIM_NEO_GPU_CUDA_INTEROP_H
#define CRESSIM_NEO_GPU_CUDA_INTEROP_H

#include "gpu/export.h"

#include <cstdint>
#include <memory>

namespace Diligent
{
struct IDeviceContext;
struct IFence;
struct IRenderDevice;
} // namespace Diligent

namespace cressim::neo::gpu
{

class SharedExportBuffer;

using CudaStreamHandle = void *;

class CRESSIM_NEO_GPU_API CudaStream
{
public:
    CudaStream();
    ~CudaStream();

    CudaStream(const CudaStream &)            = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    bool initialize();
    void reset();

    bool isInitialized() const noexcept;
    CudaStreamHandle handle() const noexcept;
    bool synchronize();
    bool copyDeviceToHostAsync(void *dst, const void *src, std::uint64_t sizeBytes);

    static bool supportsCudaInteropBuild() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class CRESSIM_NEO_GPU_API CudaExternalTimelineSemaphore
{
public:
    CudaExternalTimelineSemaphore();
    ~CudaExternalTimelineSemaphore();

    CudaExternalTimelineSemaphore(const CudaExternalTimelineSemaphore &)            = delete;
    CudaExternalTimelineSemaphore &operator=(const CudaExternalTimelineSemaphore &) = delete;

    bool initialize(Diligent::IRenderDevice *renderDevice, const char *name);
    void reset();

    bool isInitialized() const noexcept;
    bool isImportedIntoCuda() const noexcept;

    bool importIntoCuda();

    bool signalOnDeviceContext(Diligent::IDeviceContext *context, std::uint64_t value);
    bool waitOnDeviceContext(Diligent::IDeviceContext *context, std::uint64_t value);
    bool signalOnCudaStream(CudaStreamHandle stream, std::uint64_t value);
    bool waitOnCudaStream(CudaStreamHandle stream, std::uint64_t value);

    Diligent::IFence *fence() const noexcept;
    void *cudaSemaphoreHandle() const noexcept;

    static bool supportsCudaInteropBuild() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class CRESSIM_NEO_GPU_API CudaSharedBuffer
{
public:
    CudaSharedBuffer();
    ~CudaSharedBuffer();

    CudaSharedBuffer(const CudaSharedBuffer &)            = delete;
    CudaSharedBuffer &operator=(const CudaSharedBuffer &) = delete;

    bool importFromSharedExportBuffer(const SharedExportBuffer &buffer);
    void reset();

    bool isImported() const noexcept;
    void *devicePointer() const noexcept;
    std::uint64_t sizeBytes() const noexcept;
    std::int32_t deviceOrdinal() const noexcept;

    static bool supportsCudaInteropBuild() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class CRESSIM_NEO_GPU_API CudaSharedBufferBridge
{
public:
    CudaSharedBufferBridge();
    ~CudaSharedBufferBridge();

    CudaSharedBufferBridge(const CudaSharedBufferBridge &)            = delete;
    CudaSharedBufferBridge &operator=(const CudaSharedBufferBridge &) = delete;

    bool initialize(Diligent::IRenderDevice *renderDevice, const char *name);
    void reset();

    bool isInitialized() const noexcept;
    bool bindSharedBuffer(const SharedExportBuffer &buffer);
    bool synchronizeToDeviceContext(Diligent::IDeviceContext *context);
    bool synchronizeFromDeviceContext(Diligent::IDeviceContext *context);
    bool copyDeviceToHostAsync(void *dst, const void *src, std::uint64_t sizeBytes);
    bool synchronizeStream();

    CudaStreamHandle streamHandle() const noexcept;
    void *devicePointer() const noexcept;
    std::uint64_t sizeBytes() const noexcept;
    std::int32_t deviceOrdinal() const noexcept;

    static bool supportsCudaInteropBuild() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_CUDA_INTEROP_H
