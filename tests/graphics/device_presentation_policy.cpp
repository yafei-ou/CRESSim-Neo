#include "engine/runtime.h"
#include "common/logger.h"


namespace
{

using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuPresentationTargetDesc;

} // namespace

int main()
{
    {
        RuntimeConfig config{};
        config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
        config.gpuDeviceDesc.presentation.enabled = false;
        config.gpuDeviceDesc.defaultRenderTargetDesc.colorFormat = Diligent::TEX_FORMAT_UNKNOWN;

        Runtime runtime;
        if (!runtime.initialize(config))
        {
            CRESSIM_LOG_ERROR( "Runtime initialization failed for headless null backend.\n");
            return 1;
        }

        GpuDevice* device = runtime.getGpuDevice();
        if (device == nullptr)
        {
            CRESSIM_LOG_ERROR( "Graphics device not available.\n");
            runtime.shutdown();
            return 1;
        }

        GpuPresentationTargetDesc presentationDesc{};
        const bool hasPresentationDesc = device->tryGetPresentationTargetDesc(presentationDesc);
        runtime.shutdown();

        if (hasPresentationDesc)
        {
            CRESSIM_LOG_ERROR( "Headless runtime unexpectedly exposed a presentation target.\n");
            return 1;
        }
    }

    {
        RuntimeConfig config{};
        config.gpuDeviceDesc.preferredBackend = GpuBackend::Null;
        config.gpuDeviceDesc.presentation.enabled = true;

        Runtime runtime;
        if (runtime.initialize(config))
        {
            CRESSIM_LOG_ERROR( "Expected runtime initialization to fail when presentation is enabled on null backend.\n");
            runtime.shutdown();
            return 1;
        }
    }

    CRESSIM_LOG_INFO( "Device presentation policy checks passed.\n");
    return 0;
}
