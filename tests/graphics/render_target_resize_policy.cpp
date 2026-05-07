#include "engine/runtime.h"
#include "common/logger.h"


namespace
{

using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetUpdateResult;

const char* backendName(GpuBackend backend)
{
    switch (backend)
    {
    case GpuBackend::D3D12:
        return "D3D12";
    case GpuBackend::Vulkan:
        return "Vulkan";
    case GpuBackend::Null:
    default:
        return "Null";
    }
}

} // namespace

int main()
{
    const GpuBackend graphicsBackends[] = {
#if PLATFORM_WIN32
        GpuBackend::D3D12,
#endif
        GpuBackend::Vulkan,
    };

    for (const GpuBackend backend : graphicsBackends)
    {
        RuntimeConfig config{};
        config.gpuDeviceDesc.preferredBackend = backend;

        Runtime runtime;
        if (!runtime.initialize(config))
        {
            CRESSIM_LOG_WARNING("Skipping render target resize policy ", backendName(backend),
                                " checks because runtime initialization failed.\n");
            continue;
        }

        GpuDevice* device = runtime.getGpuDevice();
        if (device == nullptr)
        {
            CRESSIM_LOG_ERROR( "Graphics device not available.\n");
            runtime.shutdown();
            return 1;
        }

        GpuRenderTargetDesc desc{};
        desc.width = 640;
        desc.height = 480;
        desc.debugName = "ResizePolicy.Target";
        GpuRenderTargetHandle target = device->renderTargetSystem().createRenderTarget(desc);
        if (!device->renderTargetSystem().isValidRenderTarget(target))
        {
            CRESSIM_LOG_ERROR( "Failed to create render target.\n");
            runtime.shutdown();
            return 1;
        }

        const GpuRenderTargetUpdateResult resizeNoOp =
            device->renderTargetSystem().resizeRenderTarget(target, 640, 480);
        if (resizeNoOp != GpuRenderTargetUpdateResult::Unchanged)
        {
            CRESSIM_LOG_ERROR( "Expected unchanged result for same-size resize.\n");
            runtime.shutdown();
            return 1;
        }

        GpuRenderTargetDesc updatedDesc{};
        if (!device->renderTargetSystem().tryGetRenderTargetDesc(target, updatedDesc))
        {
            CRESSIM_LOG_ERROR( "Failed to fetch render target descriptor.\n");
            runtime.shutdown();
            return 1;
        }

        updatedDesc.debugName = "ResizePolicy.Renamed";
        const GpuRenderTargetUpdateResult metadataUpdate =
            device->renderTargetSystem().reconfigureRenderTarget(target, updatedDesc);
        if (metadataUpdate != GpuRenderTargetUpdateResult::Unchanged)
        {
            CRESSIM_LOG_ERROR( "Expected unchanged result for metadata-only reconfigure.\n");
            runtime.shutdown();
            return 1;
        }

        updatedDesc.width = 800;
        const GpuRenderTargetUpdateResult resizedUpdate =
            device->renderTargetSystem().reconfigureRenderTarget(target, updatedDesc);
        if (resizedUpdate != GpuRenderTargetUpdateResult::Recreated)
        {
            CRESSIM_LOG_ERROR( "Expected recreated result for dimension-changing reconfigure.\n");
            runtime.shutdown();
            return 1;
        }

        GpuRenderTargetDesc finalDesc{};
        if (!device->renderTargetSystem().tryGetRenderTargetDesc(target, finalDesc))
        {
            CRESSIM_LOG_ERROR( "Failed to fetch final descriptor.\n");
            runtime.shutdown();
            return 1;
        }

        runtime.shutdown();

        if (finalDesc.width != 800 || finalDesc.height != 480)
        {
            CRESSIM_LOG_ERROR( "Unexpected final descriptor dimensions.\n");
            return 1;
        }
    }

    CRESSIM_LOG_INFO( "Render target resize policy checks passed.\n");
    return 0;
}
