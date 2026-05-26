#ifndef CRESSIM_NEO_GPU_CUDA_INTEROP_H
#define CRESSIM_NEO_GPU_CUDA_INTEROP_H

#include "gpu/export.h"

#include <cstdint>
#include <memory>

namespace Diligent
{
class IDeviceContext;
class IFence;
class IRenderDevice;
} // namespace Diligent

namespace cressim::neo::gpu
{

using CudaStreamHandle = void *;

class CRESSIM_NEO_GPU_API CudaExternalTimelineSemaphore
{
public:
    CudaExternalTimelineSemaphore();
    ~CudaExternalTimelineSemaphore();

    CudaExternalTimelineSemaphore(const CudaExternalTimelineSemaphore &) = delete;
    CudaExternalTimelineSemaphore &operator=(const CudaExternalTimelineSemaphore &) = delete;

    bool initializeForVulkan(Diligent::IRenderDevice *renderDevice, const char *name);
    void reset();

    bool isInitialized() const noexcept;
    bool isImportedIntoCuda() const noexcept;

    bool importIntoCuda();

    bool signalOnDeviceContext(Diligent::IDeviceContext *context, std::uint64_t value);
    bool waitOnDeviceContext(Diligent::IDeviceContext *context, std::uint64_t value);
    bool signalOnCudaStream(CudaStreamHandle stream, std::uint64_t value);
    bool waitOnCudaStream(CudaStreamHandle stream, std::uint64_t value);

    Diligent::IFence *fence() const noexcept;

    static bool supportsCudaInteropBuild() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_CUDA_INTEROP_H
