#ifndef CRESSIM_NEO_GRAPHICS_ENVIRONMENT_IBL_BAKER_H
#define CRESSIM_NEO_GRAPHICS_ENVIRONMENT_IBL_BAKER_H

#include "graphics/export.h"
#include "graphics/render_resource_manager.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

/// @file environment_ibl_baker.h
/// @brief Offline and runtime IBL convolution baker for generating diffuse irradiance and specular
/// prefiltered cubemaps.

namespace cressim::neo::graphics
{

/// @brief In-memory floating-point image payload representing a single cubemap face.
struct EnvironmentCubemapImage
{
    std::uint32_t width  = 0u;            ///< Face width in pixels.
    std::uint32_t height = 0u;            ///< Face height in pixels.
    std::vector<Diligent::float4> pixels; ///< Array of linear RGBA float HDR pixel values.
};

/// @brief Parameter tuning descriptor for Monte Carlo irradiance and specular cubemap baking.
struct EnvironmentIblBakeOptions
{
    std::uint32_t irradianceSize = 16u; ///< Resolution of baked diffuse irradiance cubemap faces.
    std::uint32_t specularSize =
        128u; ///< Resolution of base mip level for prefiltered specular cubemap.
    std::uint32_t specularMipCount =
        7u; ///< Number of roughness mip levels generated for specular convolution.
    std::uint32_t irradianceSampleCount =
        256u; ///< Monte Carlo hemisphere sample count per texel for diffuse irradiance.
    std::uint32_t specularSampleCount =
        128u; ///< Importance sampling sample count per texel/roughness for specular reflection.
    float intensity = 1.0f; ///< Global radiant intensity multiplier.
    float backgroundIntensity =
        -1.0f; ///< Background skybox display intensity (-1 inherits `intensity`).
};

/// @brief Convolves raw in-memory cubemap face images and registers IBL textures in the resource
/// manager.
/// @param resources Resource manager to receive generated textures.
/// @param faces Array of 6 cubemap face image payloads (+X, -X, +Y, -Y, +Z, -Z).
/// @param options Baking resolution and sample count options.
/// @return Populated EnvironmentIblDesc referencing the registered background, irradiance, and
/// specular cubemaps.
/// @throws std::runtime_error if face dimensions do not match.
CRESSIM_NEO_GRAPHICS_API EnvironmentIblDesc createEnvironmentIblFromCubemapImages(
    RenderResourceManager &resources, const std::array<EnvironmentCubemapImage, 6u> &faces,
    const EnvironmentIblBakeOptions &options = {});

/// @brief Loads cubemap face image files from disk, performs IBL convolution, and registers
/// resulting textures.
/// @param resources Resource manager to receive generated textures.
/// @param facePaths Array of 6 filesystem paths to image files.
/// @param options Baking options.
/// @return Populated EnvironmentIblDesc referencing the registered cubemaps.
/// @throws std::runtime_error if an image cannot be loaded or the loaded face dimensions differ.
CRESSIM_NEO_GRAPHICS_API EnvironmentIblDesc createEnvironmentIblFromCubemapFiles(
    RenderResourceManager &resources, const std::array<std::filesystem::path, 6u> &facePaths,
    const EnvironmentIblBakeOptions &options = {});

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_ENVIRONMENT_IBL_BAKER_H
