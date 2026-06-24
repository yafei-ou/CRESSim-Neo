#include "gpu/cuda_interop.h"
#include "gpu/cuda_interop_native.h"
#include "gpu/shared_export_buffer.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "common/logger.h"

#if CRESSIM_NEO_HAS_CUDA_INTEROP
#include <cuda_runtime_api.h>
#endif

namespace cressim::neo::gpu
{

struct CudaExternalTimelineSemaphore::Impl
{
    interop::TimelineSemaphoreState nativeSemaphore;

#if CRESSIM_NEO_HAS_CUDA_INTEROP
    cudaExternalSemaphore_t cudaSemaphore = nullptr;
#endif
    bool importedIntoCuda = false;
};

struct CudaStream::Impl
{
#if CRESSIM_NEO_HAS_CUDA_INTEROP
    cudaStream_t stream = nullptr;
#endif
};

struct CudaSharedBuffer::Impl
{
#if CRESSIM_NEO_HAS_CUDA_INTEROP
    cudaExternalMemory_t cudaMemory = nullptr;
#endif
    void *devicePointer       = nullptr;
    std::uint64_t sizeInBytes = 0u;
    bool imported             = false;
};

struct CudaSharedBufferBridge::Impl
{
    CudaStream stream;
    CudaExternalTimelineSemaphore semaphore;
    CudaSharedBuffer sharedBuffer;
    Diligent::IBuffer *importedSourceBuffer = nullptr;
    std::uint64_t nextFenceValue            = 1u;
};

CudaStream::CudaStream() : mImpl{std::make_unique<Impl>()} {}

CudaStream::~CudaStream()
{
    reset();
}

bool CudaStream::initialize()
{
    reset();

#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return false;
#else
    return cudaStreamCreate(&mImpl->stream) == cudaSuccess && mImpl->stream != nullptr;
#endif
}

void CudaStream::reset()
{
    if (mImpl == nullptr)
    {
        return;
    }

#if CRESSIM_NEO_HAS_CUDA_INTEROP
    if (mImpl->stream != nullptr)
    {
        cudaStreamDestroy(mImpl->stream);
        mImpl->stream = nullptr;
    }
#endif
}

bool CudaStream::isInitialized() const noexcept
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return false;
#else
    return mImpl != nullptr && mImpl->stream != nullptr;
#endif
}

CudaStreamHandle CudaStream::handle() const noexcept
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return nullptr;
#else
    return mImpl != nullptr ? static_cast<CudaStreamHandle>(mImpl->stream) : nullptr;
#endif
}

bool CudaStream::synchronize()
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return false;
#else
    return mImpl != nullptr && mImpl->stream != nullptr &&
           cudaStreamSynchronize(mImpl->stream) == cudaSuccess;
#endif
}

bool CudaStream::copyDeviceToHostAsync(void *dst, const void *src, const std::uint64_t sizeBytes)
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    (void)dst;
    (void)src;
    (void)sizeBytes;
    return false;
#else
    return mImpl != nullptr && mImpl->stream != nullptr && dst != nullptr && src != nullptr &&
           sizeBytes > 0u &&
           cudaMemcpyAsync(dst, src, static_cast<std::size_t>(sizeBytes), cudaMemcpyDeviceToHost,
                           mImpl->stream) == cudaSuccess;
#endif
}

bool CudaStream::supportsCudaInteropBuild() noexcept
{
#if CRESSIM_NEO_HAS_CUDA_INTEROP
    return true;
#else
    return false;
#endif
}

CudaExternalTimelineSemaphore::CudaExternalTimelineSemaphore() : mImpl{std::make_unique<Impl>()} {}

CudaExternalTimelineSemaphore::~CudaExternalTimelineSemaphore()
{
    reset();
}

bool CudaExternalTimelineSemaphore::initialize(Diligent::IRenderDevice *renderDevice,
                                               const char *name)
{
    reset();

    if (renderDevice == nullptr || mImpl == nullptr)
    {
        return false;
    }

    return interop::createExportableTimelineSemaphore(renderDevice, name, mImpl->nativeSemaphore);
}

void CudaExternalTimelineSemaphore::reset()
{
    if (mImpl == nullptr)
    {
        return;
    }

#if CRESSIM_NEO_HAS_CUDA_INTEROP
    if (mImpl->importedIntoCuda && mImpl->cudaSemaphore != nullptr)
    {
        cudaDestroyExternalSemaphore(mImpl->cudaSemaphore);
        mImpl->cudaSemaphore    = nullptr;
        mImpl->importedIntoCuda = false;
    }
#endif

    interop::resetExportableTimelineSemaphore(mImpl->nativeSemaphore);
}

bool CudaExternalTimelineSemaphore::isInitialized() const noexcept
{
    return mImpl != nullptr && mImpl->nativeSemaphore.initialized &&
           mImpl->nativeSemaphore.fence != nullptr;
}

bool CudaExternalTimelineSemaphore::isImportedIntoCuda() const noexcept
{
    return mImpl != nullptr && mImpl->importedIntoCuda;
}

bool CudaExternalTimelineSemaphore::importIntoCuda()
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return false;
#else
    if (!isInitialized() || mImpl->importedIntoCuda)
    {
        return isInitialized() && mImpl->importedIntoCuda;
    }

    interop::NativeHandle nativeHandle{};
    if (!interop::exportSemaphoreHandle(mImpl->nativeSemaphore, nativeHandle) ||
        !interop::isValid(nativeHandle))
    {
        CRESSIM_LOG_WARNING(
            "CudaExternalTimelineSemaphore: failed to export native semaphore handle.");
        return false;
    }

    cudaExternalSemaphoreHandleDesc handleDesc{};
#if defined(_WIN32)
    switch (mImpl->nativeSemaphore.deviceType)
    {
    case Diligent::RENDER_DEVICE_TYPE_D3D12:
        handleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        break;
    case Diligent::RENDER_DEVICE_TYPE_VULKAN:
        handleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        break;
    default:
        interop::closeNativeHandle(nativeHandle);
        CRESSIM_LOG_WARNING(
            "CudaExternalTimelineSemaphore: unsupported Win32 semaphore device type ",
            static_cast<int>(mImpl->nativeSemaphore.deviceType), ".");
        return false;
    }
    handleDesc.handle.win32.handle = nativeHandle.win32Handle;
#elif defined(__linux__)
    handleDesc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    handleDesc.handle.fd = nativeHandle.fd;
#else
    return false;
#endif

    const cudaError_t importResult =
        cudaImportExternalSemaphore(&mImpl->cudaSemaphore, &handleDesc);
    if (importResult != cudaSuccess || mImpl->cudaSemaphore == nullptr)
    {
        interop::closeNativeHandle(nativeHandle);
        mImpl->cudaSemaphore = nullptr;
        CRESSIM_LOG_WARNING(
            "CudaExternalTimelineSemaphore: cudaImportExternalSemaphore failed with code ",
            static_cast<int>(importResult), ".");
        return false;
    }

#if defined(_WIN32)
    // CUDA retains the imported semaphore, but Win32 handle ownership stays with the caller.
    interop::closeNativeHandle(nativeHandle);
#else
    interop::releaseOwnership(nativeHandle);
#endif

    mImpl->importedIntoCuda = true;
    return true;
#endif
}

bool CudaExternalTimelineSemaphore::signalOnDeviceContext(Diligent::IDeviceContext *context,
                                                          const std::uint64_t value)
{
    if (context == nullptr || mImpl == nullptr || mImpl->nativeSemaphore.fence == nullptr)
    {
        return false;
    }

    context->EnqueueSignal(mImpl->nativeSemaphore.fence, value);
    return true;
}

bool CudaExternalTimelineSemaphore::waitOnDeviceContext(Diligent::IDeviceContext *context,
                                                        const std::uint64_t value)
{
    if (context == nullptr || mImpl == nullptr || mImpl->nativeSemaphore.fence == nullptr)
    {
        return false;
    }

    context->DeviceWaitForFence(mImpl->nativeSemaphore.fence, value);
    return true;
}

bool CudaExternalTimelineSemaphore::signalOnCudaStream(const CudaStreamHandle stream,
                                                       const std::uint64_t value)
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    (void)stream;
    (void)value;
    return false;
#else
    if (!mImpl->importedIntoCuda || mImpl->cudaSemaphore == nullptr || stream == nullptr)
    {
        return false;
    }

    cudaExternalSemaphoreSignalParams signalParams{};
    signalParams.params.fence.value = value;
    return cudaSignalExternalSemaphoresAsync(&mImpl->cudaSemaphore, &signalParams, 1u,
                                             static_cast<cudaStream_t>(stream)) == cudaSuccess;
#endif
}

bool CudaExternalTimelineSemaphore::waitOnCudaStream(const CudaStreamHandle stream,
                                                     const std::uint64_t value)
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    (void)stream;
    (void)value;
    return false;
#else
    if (!mImpl->importedIntoCuda || mImpl->cudaSemaphore == nullptr)
    {
        return false;
    }

    cudaExternalSemaphoreWaitParams waitParams{};
    waitParams.params.fence.value = value;
    return cudaWaitExternalSemaphoresAsync(&mImpl->cudaSemaphore, &waitParams, 1u,
                                           static_cast<cudaStream_t>(stream)) == cudaSuccess;
#endif
}

Diligent::IFence *CudaExternalTimelineSemaphore::fence() const noexcept
{
    return mImpl != nullptr ? mImpl->nativeSemaphore.fence.RawPtr() : nullptr;
}

void *CudaExternalTimelineSemaphore::cudaSemaphoreHandle() const noexcept
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return nullptr;
#else
    return (mImpl != nullptr && mImpl->importedIntoCuda) ? static_cast<void *>(mImpl->cudaSemaphore)
                                                         : nullptr;
#endif
}

bool CudaExternalTimelineSemaphore::supportsCudaInteropBuild() noexcept
{
#if CRESSIM_NEO_HAS_CUDA_INTEROP
    return true;
#else
    return false;
#endif
}

CudaSharedBuffer::CudaSharedBuffer() : mImpl{std::make_unique<Impl>()} {}

CudaSharedBuffer::~CudaSharedBuffer()
{
    reset();
}

bool CudaSharedBuffer::importFromSharedExportBuffer(const SharedExportBuffer &buffer)
{
    reset();

    const std::uint64_t bufferSizeInBytes = buffer.sizeBytes();
    if (!buffer.usesNativeSharedAllocation() || bufferSizeInBytes == 0u)
    {
        return false;
    }

#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return false;
#elif !defined(_WIN32) && !defined(__linux__)
    return false;
#else
    interop::NativeHandle nativeHandle{};
    if (!buffer.exportNativeHandle(nativeHandle) || !interop::isValid(nativeHandle))
    {
        CRESSIM_LOG_WARNING("CudaSharedBuffer: failed to export native buffer handle.");
        return false;
    }

    cudaExternalMemoryHandleDesc handleDesc{};
#if defined(_WIN32)
    switch (buffer.nativeRenderDeviceType())
    {
    case Diligent::RENDER_DEVICE_TYPE_D3D12:
        handleDesc.type  = cudaExternalMemoryHandleTypeD3D12Resource;
        handleDesc.flags = cudaExternalMemoryDedicated;
        break;
    case Diligent::RENDER_DEVICE_TYPE_VULKAN:
        handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        break;
    default:
        interop::closeNativeHandle(nativeHandle);
        CRESSIM_LOG_WARNING("CudaSharedBuffer: unsupported Win32 memory device type ",
                            static_cast<int>(buffer.nativeRenderDeviceType()), ".");
        return false;
    }
    handleDesc.handle.win32.handle = nativeHandle.win32Handle;
#else
    handleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    handleDesc.handle.fd = nativeHandle.fd;
#endif
    handleDesc.size = static_cast<std::size_t>(bufferSizeInBytes);

    const cudaError_t importResult = cudaImportExternalMemory(&mImpl->cudaMemory, &handleDesc);
    if (importResult != cudaSuccess || mImpl->cudaMemory == nullptr)
    {
        interop::closeNativeHandle(nativeHandle);
        mImpl->cudaMemory = nullptr;
        CRESSIM_LOG_WARNING("CudaSharedBuffer: cudaImportExternalMemory failed with code ",
                            static_cast<int>(importResult), ".");
        return false;
    }

#if defined(_WIN32)
    // CUDA retains the imported memory, but Win32 handle ownership stays with the caller.
    interop::closeNativeHandle(nativeHandle);
#else
    interop::releaseOwnership(nativeHandle);
#endif

    cudaExternalMemoryBufferDesc mapDesc{};
    mapDesc.offset = 0u;
    mapDesc.size   = static_cast<std::size_t>(bufferSizeInBytes);

    void *mappedPointer = nullptr;
    const cudaError_t mapResult =
        cudaExternalMemoryGetMappedBuffer(&mappedPointer, mImpl->cudaMemory, &mapDesc);
    if (mapResult != cudaSuccess || mappedPointer == nullptr)
    {
        cudaDestroyExternalMemory(mImpl->cudaMemory);
        mImpl->cudaMemory = nullptr;
        CRESSIM_LOG_WARNING("CudaSharedBuffer: cudaExternalMemoryGetMappedBuffer failed with code ",
                            static_cast<int>(mapResult), ".");
        return false;
    }

    mImpl->devicePointer = mappedPointer;
    mImpl->sizeInBytes   = bufferSizeInBytes;
    mImpl->imported      = true;
    return true;
#endif
}

void CudaSharedBuffer::reset()
{
    if (mImpl == nullptr)
    {
        return;
    }

#if CRESSIM_NEO_HAS_CUDA_INTEROP
    if (mImpl->cudaMemory != nullptr)
    {
        cudaDestroyExternalMemory(mImpl->cudaMemory);
        mImpl->cudaMemory = nullptr;
    }
#endif

    mImpl->devicePointer = nullptr;
    mImpl->sizeInBytes   = 0u;
    mImpl->imported      = false;
}

bool CudaSharedBuffer::isImported() const noexcept
{
    return mImpl != nullptr && mImpl->imported;
}

void *CudaSharedBuffer::devicePointer() const noexcept
{
    return mImpl != nullptr ? mImpl->devicePointer : nullptr;
}

std::uint64_t CudaSharedBuffer::sizeBytes() const noexcept
{
    return mImpl != nullptr ? mImpl->sizeInBytes : 0u;
}

std::int32_t CudaSharedBuffer::deviceOrdinal() const noexcept
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return -1;
#else
    if (mImpl == nullptr || mImpl->devicePointer == nullptr)
    {
        return -1;
    }

    cudaPointerAttributes attributes{};
    const cudaError_t result = cudaPointerGetAttributes(&attributes, mImpl->devicePointer);
    if (result != cudaSuccess)
    {
        return -1;
    }
#if CUDART_VERSION >= 10000
    return attributes.device;
#else
    return attributes.device;
#endif
#endif
}

bool CudaSharedBuffer::supportsCudaInteropBuild() noexcept
{
#if CRESSIM_NEO_HAS_CUDA_INTEROP
    return true;
#else
    return false;
#endif
}

CudaSharedBufferBridge::CudaSharedBufferBridge() : mImpl{std::make_unique<Impl>()} {}

CudaSharedBufferBridge::~CudaSharedBufferBridge()
{
    reset();
}

bool CudaSharedBufferBridge::initialize(Diligent::IRenderDevice *renderDevice, const char *name)
{
    if (mImpl == nullptr)
    {
        return false;
    }

    if (!mImpl->stream.isInitialized() && !mImpl->stream.initialize())
    {
        CRESSIM_LOG_WARNING("CudaSharedBufferBridge: failed to initialize CUDA stream.");
        return false;
    }

    if (!mImpl->semaphore.isInitialized() && !mImpl->semaphore.initialize(renderDevice, name))
    {
        CRESSIM_LOG_WARNING("CudaSharedBufferBridge: failed to create exportable semaphore.");
        return false;
    }

    if (!mImpl->semaphore.isImportedIntoCuda() && !mImpl->semaphore.importIntoCuda())
    {
        CRESSIM_LOG_WARNING("CudaSharedBufferBridge: failed to import semaphore into CUDA.");
        return false;
    }

    return true;
}

void CudaSharedBufferBridge::reset()
{
    if (mImpl == nullptr)
    {
        return;
    }

    mImpl->sharedBuffer.reset();
    mImpl->importedSourceBuffer = nullptr;
    mImpl->nextFenceValue       = 1u;
    mImpl->semaphore.reset();
    mImpl->stream.reset();
}

bool CudaSharedBufferBridge::isInitialized() const noexcept
{
    return mImpl != nullptr && mImpl->stream.isInitialized() && mImpl->semaphore.isInitialized();
}

bool CudaSharedBufferBridge::bindSharedBuffer(const SharedExportBuffer &buffer)
{
    if (mImpl == nullptr || !buffer.usesNativeSharedAllocation())
    {
        return false;
    }

    if (mImpl->importedSourceBuffer == buffer.buffer() && mImpl->sharedBuffer.isImported())
    {
        return true;
    }

    mImpl->sharedBuffer.reset();
    mImpl->importedSourceBuffer = nullptr;
    if (!mImpl->sharedBuffer.importFromSharedExportBuffer(buffer))
    {
        return false;
    }

    mImpl->importedSourceBuffer = buffer.buffer();
    return true;
}

bool CudaSharedBufferBridge::synchronizeFromDeviceContext(Diligent::IDeviceContext *context)
{
    if (mImpl == nullptr || context == nullptr || !mImpl->stream.isInitialized() ||
        !mImpl->semaphore.isInitialized() || !mImpl->semaphore.isImportedIntoCuda())
    {
        return false;
    }

    const std::uint64_t fenceValue = mImpl->nextFenceValue++;
    if (!mImpl->semaphore.signalOnDeviceContext(context, fenceValue))
    {
        return false;
    }

    context->Flush();
    // Queue the graphics/physics completion wait on the CUDA default stream so subsequent
    // default-stream tracker work and blocking CUDA streams observe the dependency without
    // a host-side stream synchronize.
    return mImpl->semaphore.waitOnCudaStream(nullptr, fenceValue);
}

bool CudaSharedBufferBridge::synchronizeToDeviceContext(Diligent::IDeviceContext *context)
{
    if (mImpl == nullptr || context == nullptr || !mImpl->stream.isInitialized() ||
        !mImpl->semaphore.isInitialized() || !mImpl->semaphore.isImportedIntoCuda())
    {
        return false;
    }

#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return false;
#else
    // Torch may have written the shared allocation on an arbitrary CUDA stream that the engine
    // does not know about. For this first prototype, conservatively synchronize the device
    // before signaling the interop semaphore back to Vulkan/D3D12.
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        return false;
    }

    const std::uint64_t fenceValue = mImpl->nextFenceValue++;
    if (!mImpl->semaphore.signalOnCudaStream(mImpl->stream.handle(), fenceValue))
    {
        return false;
    }

    return mImpl->semaphore.waitOnDeviceContext(context, fenceValue);
#endif
}

bool CudaSharedBufferBridge::copyDeviceToHostAsync(void *dst, const void *src,
                                                   const std::uint64_t sizeBytes)
{
    return mImpl != nullptr && mImpl->stream.copyDeviceToHostAsync(dst, src, sizeBytes);
}

bool CudaSharedBufferBridge::synchronizeStream()
{
    return mImpl != nullptr && mImpl->stream.synchronize();
}

CudaStreamHandle CudaSharedBufferBridge::streamHandle() const noexcept
{
    return mImpl != nullptr ? mImpl->stream.handle() : nullptr;
}

void *CudaSharedBufferBridge::devicePointer() const noexcept
{
    return mImpl != nullptr ? mImpl->sharedBuffer.devicePointer() : nullptr;
}

std::uint64_t CudaSharedBufferBridge::sizeBytes() const noexcept
{
    return mImpl != nullptr ? mImpl->sharedBuffer.sizeBytes() : 0u;
}

std::int32_t CudaSharedBufferBridge::deviceOrdinal() const noexcept
{
#if !CRESSIM_NEO_HAS_CUDA_INTEROP
    return -1;
#else
    const void *pointer = devicePointer();
    if (pointer == nullptr)
    {
        return -1;
    }

    cudaPointerAttributes attributes{};
    const cudaError_t result = cudaPointerGetAttributes(&attributes, pointer);
    if (result != cudaSuccess)
    {
        return -1;
    }
    return attributes.device;
#endif
}

bool CudaSharedBufferBridge::supportsCudaInteropBuild() noexcept
{
    return CudaStream::supportsCudaInteropBuild() &&
           CudaExternalTimelineSemaphore::supportsCudaInteropBuild() &&
           CudaSharedBuffer::supportsCudaInteropBuild();
}

} // namespace cressim::neo::gpu
