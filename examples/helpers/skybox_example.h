#ifndef CRESSIM_NEO_EXAMPLES_HELPERS_SKYBOX_EXAMPLE_H
#define CRESSIM_NEO_EXAMPLES_HELPERS_SKYBOX_EXAMPLE_H

#include "graphics/environment_ibl_baker.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentTools/TextureLoader/interface/Image.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace cressim::neo::examples::helpers
{

inline float srgbToLinear(float value)
{
    if (value <= 0.04045f)
    {
        return value / 12.92f;
    }
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

inline std::array<graphics::EnvironmentCubemapImage, 6u> loadHorizontalCrossCubemapFaces(
    const std::filesystem::path &path)
{
    Diligent::RefCntAutoPtr<Diligent::Image> image;
    Diligent::CreateImageFromFile(path.string().c_str(), &image, nullptr);
    if (image == nullptr)
    {
        throw std::runtime_error("Failed to load cubemap cross image: " + path.string());
    }

    const Diligent::ImageDesc &desc = image->GetDesc();
    if (desc.ComponentType != Diligent::VT_UINT8 ||
        (desc.NumComponents != 3u && desc.NumComponents != 4u))
    {
        throw std::runtime_error("Unsupported cubemap cross format: " + path.string());
    }
    if (desc.Width == 0u || desc.Height == 0u || image->GetData() == nullptr)
    {
        throw std::runtime_error("Cubemap cross image is empty: " + path.string());
    }
    if (desc.Width % 4u != 0u || desc.Height % 3u != 0u)
    {
        throw std::runtime_error("Cubemap cross image must be a 4x3 layout: " + path.string());
    }

    const std::uint32_t faceSizeX = desc.Width / 4u;
    const std::uint32_t faceSizeY = desc.Height / 3u;
    if (faceSizeX != faceSizeY)
    {
        throw std::runtime_error("Cubemap cross faces must be square: " + path.string());
    }

    struct FaceRect
    {
        std::uint32_t tileX;
        std::uint32_t tileY;
    };

    // Horizontal cross layout:
    //       +Y
    // -X +Z +X -Z
    //       -Y
    constexpr std::array<FaceRect, 6u> kFaceRects = {{
        {2u, 1u}, // +X
        {0u, 1u}, // -X
        {1u, 0u}, // +Y
        {1u, 2u}, // -Y
        {1u, 1u}, // +Z
        {3u, 1u}, // -Z
    }};

    const auto *src = image->GetData()->GetConstDataPtr<std::uint8_t>();
    std::array<graphics::EnvironmentCubemapImage, 6u> faces;
    for (std::uint32_t faceIndex = 0u; faceIndex < faces.size(); ++faceIndex)
    {
        auto &face = faces[faceIndex];
        face.width = faceSizeX;
        face.height = faceSizeY;
        face.pixels.resize(static_cast<std::size_t>(face.width) * face.height);

        const std::uint32_t originX = kFaceRects[faceIndex].tileX * faceSizeX;
        const std::uint32_t originY = kFaceRects[faceIndex].tileY * faceSizeY;
        for (std::uint32_t y = 0u; y < face.height; ++y)
        {
            const auto *row =
                src + static_cast<std::size_t>(originY + y) * desc.RowStride +
                static_cast<std::size_t>(originX) * desc.NumComponents;
            for (std::uint32_t x = 0u; x < face.width; ++x)
            {
                const auto *pixel = row + static_cast<std::size_t>(x) * desc.NumComponents;
                const float r = static_cast<float>(pixel[0]) / 255.0f;
                const float g = static_cast<float>(pixel[1]) / 255.0f;
                const float b = static_cast<float>(pixel[2]) / 255.0f;
                const float a =
                    desc.NumComponents == 4u ? static_cast<float>(pixel[3]) / 255.0f : 1.0f;
                face.pixels[static_cast<std::size_t>(y) * face.width + x] = {
                    srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), a};
            }
        }
    }

    return faces;
}

inline graphics::EnvironmentIblDesc createEnvironmentIblFromHorizontalCross(
    graphics::RenderResourceManager &resources, const std::filesystem::path &path,
    const graphics::EnvironmentIblBakeOptions &options = {})
{
    return graphics::createEnvironmentIblFromCubemapImages(resources,
                                                           loadHorizontalCrossCubemapFaces(path),
                                                           options);
}

inline graphics::EnvironmentIblDesc createEnvironmentIblFromExampleSkyboxFiles(
    graphics::RenderResourceManager &resources, const std::filesystem::path &skyboxDir,
    const graphics::EnvironmentIblBakeOptions &options = {})
{
    const std::array<std::filesystem::path, 6u> facePaths = {
        skyboxDir / "posx.jpg", skyboxDir / "negx.jpg", skyboxDir / "posy.jpg",
        skyboxDir / "negy.jpg", skyboxDir / "posz.jpg", skyboxDir / "negz.jpg"};
    return graphics::createEnvironmentIblFromCubemapFiles(resources, facePaths, options);
}

} // namespace cressim::neo::examples::helpers

#endif
