#ifndef CRESSIM_NEO_EXAMPLES_HELPERS_READBACK_IMAGE_IO_H
#define CRESSIM_NEO_EXAMPLES_HELPERS_READBACK_IMAGE_IO_H

#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace cressim::neo::examples::helpers
{

struct ReadbackPixel
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

inline bool isValidColorReadback(const gpu::GpuRenderTargetReadbackEvent &event)
{
    if (event.colorWidth == 0u || event.colorHeight == 0u)
    {
        return false;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    if (formatAttribs.Format == Diligent::TEX_FORMAT_UNKNOWN || formatAttribs.IsTypeless ||
        formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPRESSED)
    {
        return false;
    }

    const std::uint32_t minStride = event.colorWidth * formatAttribs.GetElementSize();
    if (event.colorRowStrideBytes < minStride)
    {
        return false;
    }

    return event.colorBytes.size() >= static_cast<std::size_t>(event.colorRowStrideBytes) *
                                          static_cast<std::size_t>(event.colorHeight);
}

inline bool isValidDepthReadback(const gpu::GpuRenderTargetReadbackEvent &event)
{
    if (event.depthWidth == 0u || event.depthHeight == 0u)
    {
        return false;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.depthFormat);
    if (formatAttribs.Format == Diligent::TEX_FORMAT_UNKNOWN || formatAttribs.IsTypeless ||
        formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPRESSED)
    {
        return false;
    }

    const std::uint32_t minStride = event.depthWidth * formatAttribs.GetElementSize();
    if (event.depthRowStrideBytes < minStride)
    {
        return false;
    }

    return event.depthBytes.size() >= static_cast<std::size_t>(event.depthRowStrideBytes) *
                                          static_cast<std::size_t>(event.depthHeight);
}

inline bool isValidLegacyColorReadback(const gpu::GpuRenderTargetReadbackEvent &event)
{
    return isValidColorReadback(event);
}

inline float halfToFloat(std::uint16_t value)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    std::uint32_t exponent   = (value >> 10u) & 0x1fu;
    std::uint32_t mantissa   = value & 0x03ffu;

    std::uint32_t bits = 0u;
    if (exponent == 0u)
    {
        if (mantissa != 0u)
        {
            exponent = 113u;
            while ((mantissa & 0x0400u) == 0u)
            {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (exponent << 23u) | (mantissa << 13u);
        }
        else
        {
            bits = sign;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline ReadbackPixel segmentationColor(std::uint32_t id)
{
    if (id == 0u)
    {
        return {};
    }

    std::uint32_t hash = id;
    hash ^= 2747636419u;
    hash *= 2654435769u;
    hash ^= hash >> 16u;
    hash *= 2654435769u;
    hash ^= hash >> 16u;
    hash *= 2654435769u;

    const float r = static_cast<float>((hash >> 0u) & 255u) / 255.0f;
    const float g = static_cast<float>((hash >> 8u) & 255u) / 255.0f;
    const float b = static_cast<float>((hash >> 16u) & 255u) / 255.0f;
    return {0.2f + 0.8f * r, 0.2f + 0.8f * g, 0.2f + 0.8f * b};
}

inline ReadbackPixel decodeColorPixel(const gpu::GpuRenderTargetReadbackEvent &event,
                                      std::uint32_t x, std::uint32_t y)
{
    ReadbackPixel pixel{};
    if (!isValidColorReadback(event))
    {
        return pixel;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    const std::size_t offset =
        static_cast<std::size_t>(y) * event.colorRowStrideBytes +
        static_cast<std::size_t>(x) * formatAttribs.GetElementSize();

    if (event.colorFormat == Diligent::TEX_FORMAT_RGBA16_FLOAT)
    {
        const std::uint16_t pixelR =
            static_cast<std::uint16_t>(event.colorBytes[offset + 0u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 1u]) << 8u);
        const std::uint16_t pixelG =
            static_cast<std::uint16_t>(event.colorBytes[offset + 2u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 3u]) << 8u);
        const std::uint16_t pixelB =
            static_cast<std::uint16_t>(event.colorBytes[offset + 4u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 5u]) << 8u);
        pixel.r = halfToFloat(pixelR);
        pixel.g = halfToFloat(pixelG);
        pixel.b = halfToFloat(pixelB);
        return pixel;
    }

    if (event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM ||
        event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM_SRGB)
    {
        pixel.r = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
        pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
        pixel.b = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
        return pixel;
    }

    if (event.colorFormat == Diligent::TEX_FORMAT_R32_UINT)
    {
        std::uint32_t id = 0u;
        std::memcpy(&id, event.colorBytes.data() + offset, sizeof(id));
        return segmentationColor(id);
    }

    pixel.r = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
    pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
    pixel.b = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
    return pixel;
}

inline float decodeDepthSample(const gpu::GpuRenderTargetReadbackEvent &event, std::uint32_t x,
                               std::uint32_t y)
{
    if (!isValidDepthReadback(event))
    {
        return 1.0f;
    }

    const auto &formatAttribs = Diligent::GetTextureFormatAttribs(event.depthFormat);
    const std::size_t offset =
        static_cast<std::size_t>(y) * event.depthRowStrideBytes +
        static_cast<std::size_t>(x) * formatAttribs.GetElementSize();

    if (event.depthFormat == Diligent::TEX_FORMAT_D32_FLOAT ||
        event.depthFormat == Diligent::TEX_FORMAT_R32_FLOAT)
    {
        float depth = 1.0f;
        std::memcpy(&depth, event.depthBytes.data() + offset, sizeof(depth));
        return depth;
    }

    if (event.depthFormat == Diligent::TEX_FORMAT_D16_UNORM)
    {
        std::uint16_t depth = 0u;
        std::memcpy(&depth, event.depthBytes.data() + offset, sizeof(depth));
        return static_cast<float>(depth) / 65535.0f;
    }

    return 1.0f;
}

inline float depthToLinear01(float depthSample, float nearClip, float farClip)
{
    if (nearClip <= 0.0f || farClip <= nearClip)
    {
        return std::clamp(depthSample, 0.0f, 1.0f);
    }
    const float zNdc = depthSample * 2.0f - 1.0f;
    const float linearDepth = (2.0f * nearClip * farClip) /
                              std::max(farClip + nearClip - zNdc * (farClip - nearClip), 1.0e-6f);
    return std::clamp((linearDepth - nearClip) / std::max(farClip - nearClip, 1.0e-6f), 0.0f,
                      1.0f);
}

inline std::uint8_t encodeByte(float value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

inline float toneMapReinhard(float value)
{
    const float clamped = std::max(value, 0.0f);
    return clamped / (1.0f + clamped);
}

inline float toneMapFilmic(float value)
{
    const float clamped = std::max(value, 0.0f);
    const float a       = 2.51f;
    const float b       = 0.03f;
    const float c       = 2.43f;
    const float d       = 0.59f;
    const float e       = 0.14f;
    return std::clamp((clamped * (a * clamped + b)) / (clamped * (c * clamped + d) + e), 0.0f,
                      1.0f);
}

inline float linearToSrgb(float value)
{
    const float clamped = std::max(value, 0.0f);
    if (clamped < 0.0031308f)
    {
        return 12.92f * clamped;
    }
    return 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
}

inline ReadbackPixel toneMapForDisplay(ReadbackPixel pixel, graphics::ToneMapper toneMapper,
                                       float exposure)
{
    pixel.r = std::max(pixel.r, 0.0f) * std::max(exposure, 0.0f);
    pixel.g = std::max(pixel.g, 0.0f) * std::max(exposure, 0.0f);
    pixel.b = std::max(pixel.b, 0.0f) * std::max(exposure, 0.0f);

    if (toneMapper == graphics::ToneMapper::Reinhard)
    {
        pixel.r = toneMapReinhard(pixel.r);
        pixel.g = toneMapReinhard(pixel.g);
        pixel.b = toneMapReinhard(pixel.b);
    }
    else if (toneMapper == graphics::ToneMapper::Filmic)
    {
        pixel.r = toneMapFilmic(pixel.r);
        pixel.g = toneMapFilmic(pixel.g);
        pixel.b = toneMapFilmic(pixel.b);
    }

    pixel.r = linearToSrgb(pixel.r);
    pixel.g = linearToSrgb(pixel.g);
    pixel.b = linearToSrgb(pixel.b);
    return pixel;
}

inline bool writeColorPpm(const std::filesystem::path &path,
                          const gpu::GpuRenderTargetReadbackEvent &event,
                          graphics::ToneMapper toneMapper, float exposure)
{
    if (!isValidColorReadback(event))
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    out << "P6\n" << event.colorWidth << " " << event.colorHeight << "\n255\n";
    std::vector<std::uint8_t> rgbRow(static_cast<std::size_t>(event.colorWidth) * 3u);
    for (std::uint32_t y = 0u; y < event.colorHeight; ++y)
    {
        for (std::uint32_t x = 0u; x < event.colorWidth; ++x)
        {
            ReadbackPixel pixel = decodeColorPixel(event, x, y);
            if (event.colorFormat == Diligent::TEX_FORMAT_RGBA16_FLOAT)
            {
                pixel = toneMapForDisplay(pixel, toneMapper, exposure);
            }
            rgbRow[static_cast<std::size_t>(x) * 3u + 0u] = encodeByte(pixel.r);
            rgbRow[static_cast<std::size_t>(x) * 3u + 1u] = encodeByte(pixel.g);
            rgbRow[static_cast<std::size_t>(x) * 3u + 2u] = encodeByte(pixel.b);
        }
        out.write(reinterpret_cast<const char *>(rgbRow.data()),
                  static_cast<std::streamsize>(rgbRow.size()));
    }
    return out.good();
}

inline bool writeColorPpm(const std::filesystem::path &path,
                          const gpu::GpuRenderTargetReadbackEvent &event)
{
    return writeColorPpm(path, event, graphics::ToneMapper::Reinhard, 1.0f);
}

inline bool writeDepthPgm(const std::filesystem::path &path,
                          const gpu::GpuRenderTargetReadbackEvent &event, float nearClip,
                          float farClip)
{
    if (!isValidDepthReadback(event))
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    out << "P5\n" << event.depthWidth << " " << event.depthHeight << "\n255\n";
    std::vector<std::uint8_t> grayRow(static_cast<std::size_t>(event.depthWidth));
    for (std::uint32_t y = 0u; y < event.depthHeight; ++y)
    {
        for (std::uint32_t x = 0u; x < event.depthWidth; ++x)
        {
            const float depth = decodeDepthSample(event, x, y);
            const float displayValue = 1.0f - depthToLinear01(depth, nearClip, farClip);
            grayRow[static_cast<std::size_t>(x)] = encodeByte(displayValue);
        }
        out.write(reinterpret_cast<const char *>(grayRow.data()),
                  static_cast<std::streamsize>(grayRow.size()));
    }
    return out.good();
}

} // namespace cressim::neo::examples::helpers

#endif // CRESSIM_NEO_EXAMPLES_HELPERS_READBACK_IMAGE_IO_H
