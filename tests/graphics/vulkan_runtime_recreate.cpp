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

bool initializeAndShutdownRuntime(std::uint64_t iteration)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        if (iteration == 0u)
        {
            CRESSIM_LOG_WARNING(
                "Skipping Vulkan runtime recreate test because runtime initialization failed.\n");
            return false;
        }

        CRESSIM_LOG_ERROR("Failed to recreate Vulkan runtime on iteration ", iteration, ".\n");
        return false;
    }

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex   = iteration;
    frame.timeSeconds  = static_cast<double>(iteration) * frame.deltaSeconds;
    runtime.tick(frame);
    runtime.shutdown();
    return true;
}

} // namespace

int main()
{
    if (!initializeAndShutdownRuntime(0u))
    {
        return 0;
    }

    for (std::uint64_t iteration = 1u; iteration < 3u; ++iteration)
    {
        if (!initializeAndShutdownRuntime(iteration))
        {
            return 1;
        }
    }

    CRESSIM_LOG_INFO("Vulkan runtime recreate checks passed.\n");
    return 0;
}
