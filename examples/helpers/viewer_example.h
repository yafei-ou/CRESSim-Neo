#ifndef CRESSIM_NEO_EXAMPLES_HELPERS_VIEWER_EXAMPLE_H
#define CRESSIM_NEO_EXAMPLES_HELPERS_VIEWER_EXAMPLE_H

#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "viewer/debug_viewer_app.h"

namespace cressim::neo::examples::helpers
{

struct ViewerExampleDefaults
{
    const char* windowTitle = "CRESSim Neo Example";
    std::uint32_t width = 640u;
    std::uint32_t height = 480u;
    bool showStats = false;
    bool vSync = false;
};

inline engine::RuntimeConfig makeRuntimeConfig(const CommonExampleOptions& options)
{
    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = options.backend;
    config.gpuDeviceDesc.enableValidation = false;
    return config;
}

inline bool windowEnabledFor(gpu::GpuBackend backend, WindowMode windowMode) noexcept
{
    if (backend == gpu::GpuBackend::Null)
    {
        return false;
    }

    if (windowMode == WindowMode::Off)
    {
        return false;
    }

    return true;
}

inline bool startFullscreenFor(WindowMode windowMode) noexcept
{
    return windowMode == WindowMode::Fullscreen;
}

inline bool startFullscreenWindowedFor(WindowMode windowMode) noexcept
{
    return windowMode == WindowMode::WindowedFullscreen;
}

inline viewer::DebugViewerAppDesc makeViewerDesc(const CommonExampleOptions& options,
                                                 const ViewerExampleDefaults& defaults)
{
    viewer::DebugViewerAppDesc desc{};
    const bool windowEnabled = windowEnabledFor(options.backend, options.windowMode);
    desc.windowTitle = defaults.windowTitle;
    desc.width = (options.windowWidth != 0u) ? options.windowWidth : defaults.width;
    desc.height = (options.windowHeight != 0u) ? options.windowHeight : defaults.height;
    desc.windowEnabled = windowEnabled;
    desc.windowVisible = windowEnabled;
    desc.startFullscreen = startFullscreenFor(options.windowMode) && windowEnabled;
    desc.startFullscreenWindowed = startFullscreenWindowedFor(options.windowMode) && windowEnabled;
    desc.vSync = defaults.vSync;
    desc.maxFrames = options.maxFrames;
    desc.showStats = defaults.showStats;
    return desc;
}

} // namespace cressim::neo::examples::helpers

#endif
