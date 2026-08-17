#ifndef CRESSIM_NEO_EXAMPLES_HELPERS_EXAMPLE_CLI_H
#define CRESSIM_NEO_EXAMPLES_HELPERS_EXAMPLE_CLI_H

#include "common/logger.h"
#include "gpu/gpu_types.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cressim::neo::examples::helpers
{

enum class WindowMode
{
    Windowed,
    WindowedFullscreen,
    Fullscreen,
    Off,
};

struct CommonExampleOptions
{
    gpu::GpuBackend backend = gpu::GpuBackend::Vulkan;
    std::uint64_t maxFrames = 0u;
    std::uint32_t envCount = 1u;
    WindowMode windowMode = WindowMode::Windowed;
    std::uint32_t windowWidth = 0u;
    std::uint32_t windowHeight = 0u;
    std::string captureVideoPath{};
    std::uint32_t captureFps = 30u;
    std::uint32_t simulationFps = 60u;
    std::uint32_t captureSwitchIntervalFrames = 0u;
};

inline gpu::GpuBackend parseBackend(const std::string& value)
{
    if (value == "null")
    {
        return gpu::GpuBackend::Null;
    }

    if (value == "d3d12")
    {
        return gpu::GpuBackend::D3D12;
    }

    if (value == "vulkan")
    {
        return gpu::GpuBackend::Vulkan;
    }

    throw std::invalid_argument("Unsupported backend: " + value);
}

inline WindowMode parseWindowMode(const std::string& value)
{
    if (value == "windowed" || value == "on" || value == "auto")
    {
        return WindowMode::Windowed;
    }

    if (value == "windowed-full" || value == "windowed-fullscreen" ||
        value == "windowed_full")
    {
        return WindowMode::WindowedFullscreen;
    }

    if (value == "full" || value == "fullscreen")
    {
        return WindowMode::Fullscreen;
    }

    if (value == "off")
    {
        return WindowMode::Off;
    }

    throw std::invalid_argument("Unsupported window mode: " + value);
}

inline std::uint32_t parseWindowDimension(const std::string& value, const char* optionName)
{
    const char* begin = value.c_str();
    char* end = nullptr;
    const auto parsed = std::strtoul(begin, &end, 10);
    if (end == begin || *end != '\0' || parsed == 0u)
    {
        throw std::invalid_argument(std::string("Invalid ") + optionName + ": " + value);
    }

    return static_cast<std::uint32_t>(parsed);
}

inline void parseWindowSize(const std::string& value, std::uint32_t& outWidth,
                            std::uint32_t& outHeight)
{
    const std::size_t split = value.find_first_of("xX");
    if (split == std::string::npos || split == 0u || split + 1u >= value.size())
    {
        throw std::invalid_argument("Invalid window size: " + value +
                                    ". Expected WIDTHxHEIGHT.");
    }

    outWidth = parseWindowDimension(value.substr(0u, split), "window width");
    outHeight = parseWindowDimension(value.substr(split + 1u), "window height");
}

inline std::uint64_t parseFrameCount(const std::string& value)
{
    const char* begin = value.c_str();
    char* end = nullptr;
    const auto parsed = std::strtoull(begin, &end, 10);
    if (end == begin || *end != '\0')
    {
        throw std::invalid_argument("Invalid frame count: " + value);
    }

    return static_cast<std::uint64_t>(parsed);
}

inline std::uint32_t parseEnvCount(const std::string& value)
{
    const char* begin = value.c_str();
    char* end = nullptr;
    const auto parsed = std::strtoul(begin, &end, 10);
    if (end == begin || *end != '\0')
    {
        throw std::invalid_argument("Invalid env count: " + value);
    }

    const auto envCount = static_cast<std::uint32_t>(parsed);
    if (envCount == 0u)
    {
        throw std::invalid_argument("--envs must be greater than zero.");
    }

    return envCount;
}

inline const char* requireOptionValue(int argc, char** argv, int& index, const char* option)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(std::string("Missing value for ") + option + ".");
    }

    return argv[++index];
}

inline bool tryParseCommonArgument(int argc, char** argv, int& index,
                                   CommonExampleOptions& options, bool includeEnvs)
{
    const std::string arg = argv[index];
    if (arg == "--backend")
    {
        options.backend = parseBackend(requireOptionValue(argc, argv, index, "--backend"));
        return true;
    }

    if (arg == "--frames")
    {
        options.maxFrames = parseFrameCount(
            requireOptionValue(argc, argv, index, "--frames"));
        return true;
    }

    if (arg == "--window")
    {
        options.windowMode = parseWindowMode(
            requireOptionValue(argc, argv, index, "--window"));
        return true;
    }

    if (arg == "--window-size")
    {
        parseWindowSize(requireOptionValue(argc, argv, index, "--window-size"),
                        options.windowWidth, options.windowHeight);
        return true;
    }

    if (arg == "--capture-video")
    {
        options.captureVideoPath = requireOptionValue(argc, argv, index, "--capture-video");
        return true;
    }

    if (arg == "--capture-fps")
    {
        options.captureFps = parseWindowDimension(
            requireOptionValue(argc, argv, index, "--capture-fps"), "capture fps");
        return true;
    }

    if (arg == "--simulation-fps")
    {
        options.simulationFps = parseWindowDimension(
            requireOptionValue(argc, argv, index, "--simulation-fps"), "simulation fps");
        return true;
    }

    if (arg == "--capture-switch-interval")
    {
        options.captureSwitchIntervalFrames = parseFrameCount(
            requireOptionValue(argc, argv, index, "--capture-switch-interval"));
        return true;
    }

    if (includeEnvs && arg == "--envs")
    {
        options.envCount = parseEnvCount(requireOptionValue(argc, argv, index, "--envs"));
        return true;
    }

    return false;
}

inline void printUsage(const char* appName, const char* extraUsage, bool includeEnvs)
{
    CRESSIM_LOG_ERROR("Usage: ", appName,
                      " [--backend vulkan|d3d12|null] [--frames N]",
                      " [--window windowed|windowed-full|full]",
                      " [--window-size WIDTHxHEIGHT]",
                      " [--capture-video PATH] [--capture-fps FPS] [--simulation-fps FPS]",
                      " [--capture-switch-interval FRAMES]",
                      includeEnvs ? " [--envs N]" : "", extraUsage != nullptr ? extraUsage : "",
                      "\n");
}

} // namespace cressim::neo::examples::helpers

#endif
