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
    Auto,
    On,
    Off,
};

struct CommonExampleOptions
{
    gpu::GpuBackend backend = gpu::GpuBackend::Vulkan;
    std::uint64_t maxFrames = 0u;
    std::uint32_t envCount = 1u;
    WindowMode windowMode = WindowMode::Auto;
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
    if (value == "auto")
    {
        return WindowMode::Auto;
    }

    if (value == "on")
    {
        return WindowMode::On;
    }

    if (value == "off")
    {
        return WindowMode::Off;
    }

    throw std::invalid_argument("Unsupported window mode: " + value);
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
                      " [--backend vulkan|d3d12|null] [--frames N] [--window on|off|auto]",
                      includeEnvs ? " [--envs N]" : "", extraUsage != nullptr ? extraUsage : "",
                      "\n");
}

} // namespace cressim::neo::examples::helpers

#endif
