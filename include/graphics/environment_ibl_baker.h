#ifndef CRESSIM_NEO_GRAPHICS_ENVIRONMENT_IBL_BAKER_H
#define CRESSIM_NEO_GRAPHICS_ENVIRONMENT_IBL_BAKER_H

#include "graphics/export.h"
#include "graphics/render_resource_manager.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace cressim::neo::graphics
{

struct EnvironmentCubemapImage
{
    std::uint32_t width  = 0u;
    std::uint32_t height = 0u;
    std::vector<Diligent::float4> pixels;
};

struct EnvironmentIblBakeOptions
{
    std::uint32_t irradianceSize        = 16u;
    std::uint32_t specularSize          = 128u;
    std::uint32_t specularMipCount      = 7u;
    std::uint32_t irradianceSampleCount = 256u;
    std::uint32_t specularSampleCount   = 128u;
    float intensity                     = 1.0f;
    // Negative values inherit `intensity` so the visible background and baked IBL
    // stay matched unless the caller explicitly wants them to differ.
    float backgroundIntensity           = -1.0f;
};

CRESSIM_NEO_GRAPHICS_API EnvironmentIblDesc createEnvironmentIblFromCubemapImages(
    RenderResourceManager &resources, const std::array<EnvironmentCubemapImage, 6u> &faces,
    const EnvironmentIblBakeOptions &options = {});

CRESSIM_NEO_GRAPHICS_API EnvironmentIblDesc createEnvironmentIblFromCubemapFiles(
    RenderResourceManager &resources, const std::array<std::filesystem::path, 6u> &facePaths,
    const EnvironmentIblBakeOptions &options = {});

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_ENVIRONMENT_IBL_BAKER_H
