#ifndef CRESSIM_NEO_SRC_ENGINE_SHARED_BUFFER_SERVICE_H
#define CRESSIM_NEO_SRC_ENGINE_SHARED_BUFFER_SERVICE_H

#include "engine/shared_buffer.h"

#include "gpu/cuda_interop.h"
#include "gpu/shared_export_buffer.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cressim::neo::gpu
{
class GpuDevice;
}

namespace cressim::neo::engine
{

class SharedBufferService
{
public:
    explicit SharedBufferService(gpu::GpuDevice &device);
    ~SharedBufferService();

    SharedBufferHandle createBuffer(const SharedBufferDesc &desc);
    bool destroyBuffer(SharedBufferHandle handle);
    void clear();

    std::vector<SharedBufferInfo> listBuffers() const;
    bool tryGetBufferInfo(SharedBufferHandle handle, SharedBufferInfo &outInfo) const;
    bool tryGetCudaView(SharedBufferHandle handle, SharedBufferCudaView &outView) const;
    bool synchronizeToCuda(SharedBufferHandle handle, Diligent::IDeviceContext *context);
    bool synchronizeFromCuda(SharedBufferHandle handle, Diligent::IDeviceContext *context);
    Diligent::IBuffer *tryGetBuffer(SharedBufferHandle handle) const noexcept;
    bool isAccessCompatible(SharedBufferHandle handle, SharedBufferAccess access) const noexcept;

private:
    struct BufferState
    {
        SharedBufferInfo info{};
        gpu::SharedExportBuffer sharedBuffer{};
        gpu::CudaSharedBufferBridge cudaBridge{};
    };

    static Diligent::BIND_FLAGS toDiligentBindFlags(SharedBufferBindFlags flags) noexcept;
    static SharedBufferAccess accessFromBindFlags(SharedBufferBindFlags flags) noexcept;

private:
    gpu::GpuDevice &mDevice;
    std::uint64_t mNextBufferId = 1u;
    std::unordered_map<std::uint64_t, std::shared_ptr<BufferState>> mBuffers;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_SRC_ENGINE_SHARED_BUFFER_SERVICE_H
