#include "gpu/shared_export_buffer.h"

#include "common/logger.h"
#include "gpu/cuda_interop_native.h"
#include "gpu/gpu_buffer_utils.h"

#include <algorithm>

namespace cressim::neo::gpu
{

struct SharedExportBuffer::Impl
{
    interop::SharedBufferState nativeSharedState;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> buffer;
    std::uint32_t capacity          = 0u;
    std::uint32_t elementStride     = 0u;
    bool exportable                 = false;
    bool usesNativeSharedAllocation = false;
};

SharedExportBuffer::SharedExportBuffer() : mImpl{std::make_unique<Impl>()} {}

SharedExportBuffer::~SharedExportBuffer()
{
    reset();
}

bool SharedExportBuffer::ensureStructuredBuffer(
    Diligent::IRenderDevice *renderDevice, const char *name, const std::uint32_t elementStride,
    const std::uint32_t requiredElementCount, const std::uint32_t minimumCapacity,
    const Diligent::BIND_FLAGS bindFlags, const Diligent::USAGE usage,
    const Diligent::CPU_ACCESS_FLAGS cpuAccess, const Diligent::Uint64 immediateContextMask,
    const std::uint32_t *queueFamilyIndices, const std::uint32_t queueFamilyIndexCount)
{
    if (renderDevice == nullptr || elementStride == 0u)
    {
        return false;
    }

    const std::uint32_t requiredCapacity =
        std::max(requiredElementCount, std::max(minimumCapacity, 1u));
    if (mImpl->buffer != nullptr && mImpl->capacity >= requiredCapacity &&
        mImpl->elementStride == elementStride)
    {
        return true;
    }

    reset();

    if (mImpl != nullptr &&
        interop::canUseExportableStructuredBuffer(renderDevice, usage, cpuAccess,
                                                  immediateContextMask, queueFamilyIndexCount) &&
        interop::createExportableStructuredBuffer(
            renderDevice, name, elementStride, requiredCapacity, bindFlags, usage, cpuAccess,
            immediateContextMask, queueFamilyIndices, queueFamilyIndexCount,
            mImpl->nativeSharedState, mImpl->buffer))
    {
        mImpl->capacity                   = requiredCapacity;
        mImpl->elementStride              = elementStride;
        mImpl->exportable                 = true;
        mImpl->usesNativeSharedAllocation = true;
        return true;
    }

    if (interop::canUseExportableStructuredBuffer(renderDevice, usage, cpuAccess,
                                                  immediateContextMask, queueFamilyIndexCount))
    {
        CRESSIM_LOG_WARNING("Falling back to a non-exportable Diligent buffer for '", name,
                            "' after native shared-buffer allocation failed.");
    }

    std::uint32_t capacity = 0u;
    if (!detail::ensureStructuredBufferCapacity(renderDevice, name, elementStride, requiredCapacity,
                                                requiredCapacity, bindFlags, usage, cpuAccess,
                                                immediateContextMask, mImpl->buffer, capacity) ||
        mImpl->buffer == nullptr)
    {
        reset();
        return false;
    }

    mImpl->capacity                   = capacity;
    mImpl->elementStride              = elementStride;
    mImpl->exportable                 = false;
    mImpl->usesNativeSharedAllocation = false;
    return true;
}

void SharedExportBuffer::reset()
{
    mImpl->buffer = nullptr;
    if (mImpl != nullptr)
    {
        interop::resetExportableStructuredBuffer(mImpl->nativeSharedState);
    }
    mImpl->capacity                   = 0u;
    mImpl->elementStride              = 0u;
    mImpl->exportable                 = false;
    mImpl->usesNativeSharedAllocation = false;
}

Diligent::IBuffer *SharedExportBuffer::buffer() const noexcept
{
    return mImpl->buffer.RawPtr();
}

const Diligent::RefCntAutoPtr<Diligent::IBuffer> &SharedExportBuffer::bufferRef() const noexcept
{
    return mImpl->buffer;
}

std::uint32_t SharedExportBuffer::capacity() const noexcept
{
    return mImpl->capacity;
}

std::uint32_t SharedExportBuffer::elementStride() const noexcept
{
    return mImpl->elementStride;
}

std::uint64_t SharedExportBuffer::sizeBytes() const noexcept
{
    return static_cast<std::uint64_t>(mImpl->capacity) * mImpl->elementStride;
}

bool SharedExportBuffer::isExportable() const noexcept
{
    return mImpl->exportable;
}

bool SharedExportBuffer::usesNativeSharedAllocation() const noexcept
{
    return mImpl->usesNativeSharedAllocation;
}

bool SharedExportBuffer::exportNativeHandle(interop::NativeHandle &outHandle) const noexcept
{
    if (mImpl == nullptr)
    {
        interop::releaseOwnership(outHandle);
        return false;
    }

    return interop::exportBufferHandle(mImpl->nativeSharedState, outHandle);
}

Diligent::RENDER_DEVICE_TYPE SharedExportBuffer::nativeRenderDeviceType() const noexcept
{
    return mImpl != nullptr ? mImpl->nativeSharedState.deviceType
                            : Diligent::RENDER_DEVICE_TYPE_UNDEFINED;
}

} // namespace cressim::neo::gpu
