#include "engine/shared_buffer_service.h"

#include "common/logger.h"
#include "gpu/gpu_device.h"
#include "gpu/gpu_types.h"

#include <algorithm>

namespace cressim::neo::engine
{

SharedBufferService::SharedBufferService(gpu::GpuDevice &device) : mDevice(device) {}

SharedBufferService::~SharedBufferService() = default;

SharedBufferHandle SharedBufferService::createBuffer(const SharedBufferDesc &desc)
{
    SharedBufferHandle handle{};
    if (desc.elementStrideBytes == 0u || desc.elementCount == 0u)
    {
        CRESSIM_LOG_ERROR("SharedBufferService: element stride and count must be non-zero.");
        return handle;
    }
    if (desc.bindFlags == SharedBufferBindFlags::None)
    {
        CRESSIM_LOG_ERROR("SharedBufferService: bindFlags must not be None.");
        return handle;
    }
    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.renderDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("SharedBufferService: physics compute backend is unavailable.");
        return handle;
    }

    auto state                     = std::make_shared<BufferState>();
    state->info.debugName          = desc.debugName;
    state->info.elementStrideBytes = desc.elementStrideBytes;
    state->info.elementCount       = desc.elementCount;
    state->info.access             = desc.access;
    state->info.bindFlags          = desc.bindFlags;

    const std::string debugName =
        desc.debugName.empty() ? "CRESSimNeo.SharedBuffer" : desc.debugName;
    if (!state->sharedBuffer.ensureStructuredBuffer(
            computeBackend.renderDevice, debugName.c_str(), desc.elementStrideBytes,
            desc.elementCount, desc.minimumCapacity, toDiligentBindFlags(desc.bindFlags),
            Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
            gpu::contextMaskForId(computeBackend.contextId)))
    {
        CRESSIM_LOG_ERROR("SharedBufferService: failed to create shared buffer '", debugName, "'.");
        return handle;
    }

    const bool supportsCudaInterop = gpu::CudaSharedBuffer::supportsCudaInteropBuild();
    if (!supportsCudaInterop)
    {
        CRESSIM_LOG_INFO("SharedBufferService: CUDA interop is unavailable in this build; shared "
                         "buffer '",
                         debugName, "' will remain engine-only.");
    }
    else if (!state->sharedBuffer.usesNativeSharedAllocation())
    {
        CRESSIM_LOG_INFO("SharedBufferService: shared buffer '", debugName,
                         "' uses a non-exportable GPU allocation and cannot be imported into "
                         "CUDA.");
    }
    else if (!state->cudaBridge.initialize(computeBackend.renderDevice, debugName.c_str()) ||
             !state->cudaBridge.bindSharedBuffer(state->sharedBuffer))
    {
        CRESSIM_LOG_WARNING("SharedBufferService: failed to import shared buffer '", debugName,
                            "' into CUDA; continuing with engine-only access.");
    }

    handle.id                    = mNextBufferId++;
    state->info.handle           = handle;
    state->info.capacity         = state->sharedBuffer.capacity();
    state->info.sizeBytes        = state->sharedBuffer.sizeBytes();
    state->info.exportable       = state->sharedBuffer.isExportable();
    state->info.importedIntoCuda = state->cudaBridge.isInitialized();
    mBuffers.emplace(handle.id, std::move(state));
    return handle;
}

bool SharedBufferService::destroyBuffer(const SharedBufferHandle handle)
{
    const auto it = mBuffers.find(handle.id);
    if (it == mBuffers.end())
    {
        return false;
    }
    if (it->second->cudaWaitPending && !flushPendingCudaWaits())
    {
        return false;
    }

    mBuffers.erase(it);
    return true;
}

void SharedBufferService::clear()
{
    flushPendingCudaWaits();
    mBuffers.clear();
    mNextBufferId = 1u;
}

std::vector<SharedBufferInfo> SharedBufferService::listBuffers() const
{
    std::vector<SharedBufferInfo> infos;
    infos.reserve(mBuffers.size());
    for (const auto &entry : mBuffers)
    {
        infos.push_back(entry.second->info);
    }
    return infos;
}

bool SharedBufferService::tryGetBufferInfo(const SharedBufferHandle handle,
                                           SharedBufferInfo &outInfo) const
{
    const auto it = mBuffers.find(handle.id);
    if (it == mBuffers.end())
    {
        return false;
    }
    outInfo = it->second->info;
    return true;
}

bool SharedBufferService::tryGetCudaView(const SharedBufferHandle handle,
                                         SharedBufferCudaView &outView) const
{
    outView       = {};
    const auto it = mBuffers.find(handle.id);
    if (it == mBuffers.end() || !it->second->cudaBridge.isInitialized())
    {
        return false;
    }

    outView.devicePointer = it->second->cudaBridge.devicePointer();
    outView.sizeBytes     = it->second->cudaBridge.sizeBytes();
    outView.deviceOrdinal = it->second->cudaBridge.deviceOrdinal();
    return outView.isValid();
}

std::shared_ptr<void> SharedBufferService::retainBuffer(const SharedBufferHandle handle) const
{
    const auto it = mBuffers.find(handle.id);
    return it != mBuffers.end() ? std::static_pointer_cast<void>(it->second) : nullptr;
}

Diligent::IBuffer *SharedBufferService::tryGetBuffer(const SharedBufferHandle handle) const noexcept
{
    const auto it = mBuffers.find(handle.id);
    return it != mBuffers.end() ? it->second->sharedBuffer.buffer() : nullptr;
}

bool SharedBufferService::synchronizeToCuda(const SharedBufferHandle handle,
                                            Diligent::IDeviceContext *context)
{
    const auto it = mBuffers.find(handle.id);
    if (it == mBuffers.end())
    {
        return false;
    }
    if (!it->second->cudaBridge.isInitialized())
    {
        CRESSIM_LOG_ERROR("SharedBufferService: shared buffer '", it->second->info.debugName,
                          "' is not imported into CUDA.");
        return false;
    }
    return it->second->cudaBridge.synchronizeFromDeviceContext(context);
}

bool SharedBufferService::synchronizeFromCuda(const SharedBufferHandle handle,
                                              Diligent::IDeviceContext *context)
{
    const auto it = mBuffers.find(handle.id);
    if (it == mBuffers.end())
    {
        return false;
    }
    if (!it->second->cudaBridge.isInitialized())
    {
        CRESSIM_LOG_ERROR("SharedBufferService: shared buffer '", it->second->info.debugName,
                          "' is not imported into CUDA.");
        return false;
    }
    const bool synchronized = it->second->cudaBridge.synchronizeToDeviceContext(context);
    it->second->cudaWaitPending = it->second->cudaWaitPending || synchronized;
    return synchronized;
}

bool SharedBufferService::flushPendingCudaWaits() noexcept
{
    const bool hasPendingWait = std::any_of(
        mBuffers.begin(), mBuffers.end(),
        [](const auto &entry) { return entry.second->cudaWaitPending; });
    if (!hasPendingWait)
    {
        return true;
    }

    gpu::GpuComputeBackendContext computeBackend{};
    if (!mDevice.tryGetPhysicsBackendContext(computeBackend) ||
        computeBackend.computeContext == nullptr)
    {
        CRESSIM_LOG_ERROR("SharedBufferService: cannot submit pending CUDA synchronization before "
                          "destroying shared buffers.");
        return false;
    }

    computeBackend.computeContext->Flush();
    for (const auto &entry : mBuffers)
    {
        entry.second->cudaWaitPending = false;
    }
    return true;
}

bool SharedBufferService::isAccessCompatible(const SharedBufferHandle handle,
                                             const SharedBufferAccess access) const noexcept
{
    const auto it = mBuffers.find(handle.id);
    if (it == mBuffers.end())
    {
        return false;
    }

    switch (it->second->info.access)
    {
    case SharedBufferAccess::ReadOnly:
        return access == SharedBufferAccess::ReadOnly;
    case SharedBufferAccess::WriteOnly:
        return access == SharedBufferAccess::WriteOnly;
    case SharedBufferAccess::ReadWrite:
        return true;
    }
    return false;
}

Diligent::BIND_FLAGS SharedBufferService::toDiligentBindFlags(
    const SharedBufferBindFlags flags) noexcept
{
    Diligent::BIND_FLAGS bindFlags = Diligent::BIND_NONE;
    if ((flags & SharedBufferBindFlags::ShaderResource) != SharedBufferBindFlags::None)
    {
        bindFlags |= Diligent::BIND_SHADER_RESOURCE;
    }
    if ((flags & SharedBufferBindFlags::UnorderedAccess) != SharedBufferBindFlags::None)
    {
        bindFlags |= Diligent::BIND_UNORDERED_ACCESS;
    }
    return bindFlags;
}

SharedBufferAccess SharedBufferService::accessFromBindFlags(
    const SharedBufferBindFlags flags) noexcept
{
    const bool hasSrv =
        (flags & SharedBufferBindFlags::ShaderResource) != SharedBufferBindFlags::None;
    const bool hasUav =
        (flags & SharedBufferBindFlags::UnorderedAccess) != SharedBufferBindFlags::None;
    if (hasSrv && hasUav)
    {
        return SharedBufferAccess::ReadWrite;
    }
    return hasUav ? SharedBufferAccess::WriteOnly : SharedBufferAccess::ReadOnly;
}

} // namespace cressim::neo::engine
