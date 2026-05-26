#include "gpu/cuda_interop.h"

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

CudaExternalTimelineSemaphore::CudaExternalTimelineSemaphore() :
    mImpl{std::make_unique<Impl>()}
{
}

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

    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{
        renderDevice, Diligent::IID_RenderDeviceVk};
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

    mImpl->vkSemaphore   = VK_NULL_HANDLE;
    mImpl->renderDevice  = nullptr;
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
    Diligent::RefCntAutoPtr<Diligent::IRenderDeviceVk> renderDeviceVk{
        mImpl->renderDevice, Diligent::IID_RenderDeviceVk};
    if (renderDeviceVk == nullptr)
    {
        return false;
    }

    const VkDevice vkDevice = renderDeviceVk->GetVkDevice();
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
    handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
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
    if (!mImpl->importedIntoCuda || mImpl->cudaSemaphore == nullptr || stream == nullptr)
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

bool CudaExternalTimelineSemaphore::supportsCudaInteropBuild() noexcept
{
#if CRESSIM_NEO_HAS_CUDA_INTEROP
    return true;
#else
    return false;
#endif
}

} // namespace cressim::neo::gpu
