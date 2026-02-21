#include "engine/runtime.h"

#include <iostream>

namespace
{

using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetUpdateResult;

} // namespace

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        std::cerr << "Runtime initialization failed.\n";
        return 1;
    }

    GpuDevice* device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        std::cerr << "Graphics device not available.\n";
        runtime.shutdown();
        return 1;
    }

    GpuRenderTargetDesc desc{};
    desc.width = 640;
    desc.height = 480;
    desc.debugName = "ResizePolicy.Target";
    GpuRenderTargetHandle target = device->createRenderTarget(desc);
    if (!device->isValidRenderTarget(target))
    {
        std::cerr << "Failed to create render target.\n";
        runtime.shutdown();
        return 1;
    }

    const GpuRenderTargetUpdateResult resizeNoOp = device->resizeRenderTarget(target, 640, 480);
    if (resizeNoOp != GpuRenderTargetUpdateResult::Unchanged)
    {
        std::cerr << "Expected unchanged result for same-size resize.\n";
        runtime.shutdown();
        return 1;
    }

    GpuRenderTargetDesc updatedDesc{};
    if (!device->tryGetRenderTargetDesc(target, updatedDesc))
    {
        std::cerr << "Failed to fetch render target descriptor.\n";
        runtime.shutdown();
        return 1;
    }

    updatedDesc.debugName = "ResizePolicy.Renamed";
    const GpuRenderTargetUpdateResult metadataUpdate = device->reconfigureRenderTarget(target, updatedDesc);
    if (metadataUpdate != GpuRenderTargetUpdateResult::Unchanged)
    {
        std::cerr << "Expected unchanged result for metadata-only reconfigure.\n";
        runtime.shutdown();
        return 1;
    }

    updatedDesc.width = 800;
    const GpuRenderTargetUpdateResult resizedUpdate = device->reconfigureRenderTarget(target, updatedDesc);
    if (resizedUpdate != GpuRenderTargetUpdateResult::Recreated)
    {
        std::cerr << "Expected recreated result for dimension-changing reconfigure.\n";
        runtime.shutdown();
        return 1;
    }

    GpuRenderTargetDesc finalDesc{};
    if (!device->tryGetRenderTargetDesc(target, finalDesc))
    {
        std::cerr << "Failed to fetch final descriptor.\n";
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();

    if (finalDesc.width != 800 || finalDesc.height != 480)
    {
        std::cerr << "Unexpected final descriptor dimensions.\n";
        return 1;
    }

    std::cout << "Render target resize policy checks passed.\n";
    return 0;
}
