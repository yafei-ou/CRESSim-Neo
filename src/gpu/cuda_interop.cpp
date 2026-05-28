#include "gpu/cuda_interop.h"
#include "gpu/shared_export_buffer.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/include/VulkanUtilities/VulkanHeaders.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/RenderDeviceVk.h"

#if CRESSIM_NEO_HAS_CUDA_INTEROP
#include <cuda_runtime_api.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

namespace cressim::neo::gpu
{

namespace
{

bool closeNativeHandle(int fd) noexcept
{
#if defined(__linux__)
    return fd < 0 || close(fd) == 0;
#else
    (void)fd;
    return true;
#endif
}

} // namespace

struct CudaExternalTimelineSemaphore::Impl
{
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> renderDevice;
    Diligent::RefCntAutoPtr<Diligent::IFence> fence;
    VkSemaphore vkSemaphore = VK_NULL_HANDLE;

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

bool CudaExternalTimelineSemaphore::initializeForVulkan(Diligent::IRenderDevice *renderDevice,
                                                        const char *name)
{
    reset();

    if (renderDevice == nullptr ||
        renderDevice->GetDeviceInfo().Type != Diligent::RENDER_DEVICE_TYPE_VULKAN)
    {
        return false;
    }

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{renderDevice,
                                                                      Diligent::IID_RenderDeviceVk};
    if (renderDeviceVk == nullptr)
    {
        return false;
    }

    const VkDevice vkDevice = renderDeviceVk->GetVkDevice();
    if (vkDevice == VK_NULL_HANDLE)
    {
        return false;
    }

    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue  = 0u;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &timelineInfo;

#if CRESSIM_NEO_HAS_CUDA_INTEROP
#if defined(__linux__)
    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    semaphoreInfo.pNext    = &exportInfo;
    exportInfo.pNext       = &timelineInfo;
#endif
#endif

    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS ||
        semaphore == VK_NULL_HANDLE)
    {
        return false;
    }

    Diligent::FenceDesc fenceDesc{};
    fenceDesc.Name = name;
    fenceDesc.Type = Diligent::FENCE_TYPE_GENERAL;

    Diligent::RefCntAutoPtr<Diligent::IFence> fence;
    renderDeviceVk->CreateFenceFromVulkanResource(semaphore, fenceDesc, &fence);
    if (fence == nullptr)
    {
        vkDestroySemaphore(vkDevice, semaphore, nullptr);
        return false;
    }

    mImpl->renderDevice = renderDevice;
    mImpl->fence        = std::move(fence);
    mImpl->vkSemaphore  = semaphore;
    return true;
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

    mImpl->fence = nullptr;

    if (mImpl->renderDevice != nullptr && mImpl->vkSemaphore != VK_NULL_HANDLE)
    {
        mImpl->renderDevice->IdleGPU();

        Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{
            mImpl->renderDevice, Diligent::IID_RenderDeviceVk};
        if (renderDeviceVk != nullptr && renderDeviceVk->GetVkDevice() != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(renderDeviceVk->GetVkDevice(), mImpl->vkSemaphore, nullptr);
        }
    }

    mImpl->vkSemaphore  = VK_NULL_HANDLE;
    mImpl->renderDevice = nullptr;
}

bool CudaExternalTimelineSemaphore::isInitialized() const noexcept
{
    return mImpl != nullptr && mImpl->fence != nullptr && mImpl->vkSemaphore != VK_NULL_HANDLE;
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

#if !defined(__linux__)
    return false;
#else
    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{mImpl->renderDevice,
                                                                      Diligent::IID_RenderDeviceVk};
    if (renderDeviceVk == nullptr)
    {
        return false;
    }

    const VkDevice vkDevice   = renderDeviceVk->GetVkDevice();
    const auto getSemaphoreFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(vkDevice, "vkGetSemaphoreFdKHR"));
    if (getSemaphoreFd == nullptr)
    {
        return false;
    }

    VkSemaphoreGetFdInfoKHR fdInfo{};
    fdInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fdInfo.semaphore  = mImpl->vkSemaphore;
    fdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int semaphoreFd = -1;
    if (getSemaphoreFd(vkDevice, &fdInfo, &semaphoreFd) != VK_SUCCESS || semaphoreFd < 0)
    {
        return false;
    }

    cudaExternalSemaphoreHandleDesc handleDesc{};
    handleDesc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    handleDesc.handle.fd = semaphoreFd;

    const cudaError_t importResult =
        cudaImportExternalSemaphore(&mImpl->cudaSemaphore, &handleDesc);
    if (importResult != cudaSuccess || mImpl->cudaSemaphore == nullptr)
    {
        closeNativeHandle(semaphoreFd);
        mImpl->cudaSemaphore = nullptr;
        return false;
    }

    mImpl->importedIntoCuda = true;
    return true;
#endif
#endif
}

bool CudaExternalTimelineSemaphore::signalOnDeviceContext(Diligent::IDeviceContext *context,
                                                          const std::uint64_t value)
{
    if (context == nullptr || mImpl == nullptr || mImpl->fence == nullptr)
    {
        return false;
    }

    context->EnqueueSignal(mImpl->fence, value);
    return true;
}

bool CudaExternalTimelineSemaphore::waitOnDeviceContext(Diligent::IDeviceContext *context,
                                                        const std::uint64_t value)
{
    if (context == nullptr || mImpl == nullptr || mImpl->fence == nullptr)
    {
        return false;
    }

    context->DeviceWaitForFence(mImpl->fence, value);
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
    return mImpl != nullptr ? mImpl->fence.RawPtr() : nullptr;
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
#elif !defined(__linux__)
    return false;
#else
    int memoryFd = -1;
    if (!buffer.exportOpaqueFd(memoryFd) || memoryFd < 0)
    {
        return false;
    }

    cudaExternalMemoryHandleDesc handleDesc{};
    handleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    handleDesc.handle.fd = memoryFd;
    handleDesc.size      = static_cast<std::size_t>(bufferSizeInBytes);

    const cudaError_t importResult = cudaImportExternalMemory(&mImpl->cudaMemory, &handleDesc);
    if (importResult != cudaSuccess || mImpl->cudaMemory == nullptr)
    {
        closeNativeHandle(memoryFd);
        mImpl->cudaMemory = nullptr;
        return false;
    }

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

bool CudaSharedBufferBridge::initializeForVulkan(Diligent::IRenderDevice *renderDevice,
                                                 const char *name)
{
    if (mImpl == nullptr)
    {
        return false;
    }

    if (!mImpl->stream.isInitialized() && !mImpl->stream.initialize())
    {
        return false;
    }

    if (!mImpl->semaphore.isInitialized() &&
        !mImpl->semaphore.initializeForVulkan(renderDevice, name))
    {
        return false;
    }

    if (!mImpl->semaphore.isImportedIntoCuda() && !mImpl->semaphore.importIntoCuda())
    {
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

bool CudaSharedBufferBridge::supportsCudaInteropBuild() noexcept
{
    return CudaStream::supportsCudaInteropBuild() &&
           CudaExternalTimelineSemaphore::supportsCudaInteropBuild() &&
           CudaSharedBuffer::supportsCudaInteropBuild();
}

} // namespace cressim::neo::gpu
