#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/runtime.h"

#include <cstdint>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::gpu::GpuBackend;

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

bool initializeAndShutdownRuntime(GpuBackend backend, std::uint64_t iteration)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = backend;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        if (iteration == 0u)
        {
            CRESSIM_LOG_WARNING("Skipping ", backendName(backend),
                                " runtime recreate test because runtime initialization failed.\n");
            return false;
        }

        CRESSIM_LOG_ERROR("Failed to recreate ", backendName(backend), " runtime on iteration ",
                          iteration, ".\n");
        return false;
    }

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex   = iteration;
    frame.timeSeconds  = static_cast<double>(iteration) * frame.deltaSeconds;
    runtime.prepare();
    const bool physicsStepSucceeded = runtime.stepPhysics(frame);
    if (physicsStepSucceeded)
    {
        (void)runtime.stepSimulationSensors(frame);
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);
    runtime.shutdown();
    return true;
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
        if (!initializeAndShutdownRuntime(backend, 0u))
        {
            continue;
        }

        for (std::uint64_t iteration = 1u; iteration < 3u; ++iteration)
        {
            if (!initializeAndShutdownRuntime(backend, iteration))
            {
                return 1;
            }
        }
    }

    CRESSIM_LOG_INFO("Runtime recreate checks passed.\n");
    return 0;
}
