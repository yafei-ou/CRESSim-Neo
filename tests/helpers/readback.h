#ifndef CRESSIM_NEO_TESTS_HELPERS_READBACK_H
#define CRESSIM_NEO_TESTS_HELPERS_READBACK_H

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace cressim::neo::tests::helpers
{

struct ReadbackPixel
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};

template <typename ReadbackEvent>
inline bool isValidReadback(const ReadbackEvent& event)
{
    if (event.width == 0u || event.height == 0u)
    {
        return false;
    }

    const auto& formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    if (formatAttribs.Format == Diligent::TEX_FORMAT_UNKNOWN || formatAttribs.IsTypeless ||
        formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPRESSED)
    {
        return false;
    }

    const std::uint32_t minStride = event.width * formatAttribs.GetElementSize();
    if (event.rowStrideBytes < minStride)
    {
        return false;
    }

    return event.colorBytes.size() >=
           static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height);
}

template <typename ReadbackEvent>
inline bool isValidDepthReadback(const ReadbackEvent& event)
{
    if (event.depthWidth == 0u || event.depthHeight == 0u)
    {
        return false;
    }

    const auto& formatAttribs = Diligent::GetTextureFormatAttribs(event.depthFormat);
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

    return event.depthBytes.size() >=
           static_cast<std::size_t>(event.depthRowStrideBytes) *
               static_cast<std::size_t>(event.depthHeight);
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

template <typename ReadbackEvent>
inline float decodeDepthValue(const ReadbackEvent& event, std::uint32_t x, std::uint32_t y)
{
    if (!isValidDepthReadback(event))
    {
        return 0.0f;
    }

    const std::size_t offset =
        static_cast<std::size_t>(y) * event.depthRowStrideBytes +
        static_cast<std::size_t>(x) *
            Diligent::GetTextureFormatAttribs(event.depthFormat).GetElementSize();
    if (event.depthFormat == Diligent::TEX_FORMAT_D32_FLOAT)
    {
        float depth = 0.0f;
        std::memcpy(&depth, event.depthBytes.data() + offset, sizeof(depth));
        return depth;
    }
    if (event.depthFormat == Diligent::TEX_FORMAT_D16_UNORM)
    {
        const std::uint16_t raw = static_cast<std::uint16_t>(event.depthBytes[offset + 0u]) |
                                  (static_cast<std::uint16_t>(event.depthBytes[offset + 1u]) << 8u);
        return static_cast<float>(raw) / 65535.0f;
    }
    return 0.0f;
}

template <typename ReadbackEvent>
inline std::size_t pixelOffset(const ReadbackEvent& event, std::uint32_t x, std::uint32_t y)
{
    return static_cast<std::size_t>(y) * event.rowStrideBytes +
           static_cast<std::size_t>(x) *
               Diligent::GetTextureFormatAttribs(event.colorFormat).GetElementSize();
}

template <typename ReadbackEvent>
inline ReadbackPixel decodePixel(const ReadbackEvent& event, std::uint32_t x, std::uint32_t y)
{
    ReadbackPixel pixel{};
    if (!isValidReadback(event))
    {
        return pixel;
    }

    const std::size_t offset = pixelOffset(event, x, y);
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
        const std::uint16_t pixelA =
            static_cast<std::uint16_t>(event.colorBytes[offset + 6u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 7u]) << 8u);
        pixel.r = halfToFloat(pixelR);
        pixel.g = halfToFloat(pixelG);
        pixel.b = halfToFloat(pixelB);
        pixel.a = halfToFloat(pixelA);
        return pixel;
    }

    if (event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM ||
        event.colorFormat == Diligent::TEX_FORMAT_BGRA8_UNORM_SRGB)
    {
        pixel.r = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
        pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
        pixel.b = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
        pixel.a = static_cast<float>(event.colorBytes[offset + 3u]) / 255.0f;
        return pixel;
    }

    pixel.r = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
    pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
    pixel.b = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
    pixel.a = static_cast<float>(event.colorBytes[offset + 3u]) / 255.0f;
    return pixel;
}

template <typename ReadbackEvent>
inline ReadbackPixel readCenterPixel(const ReadbackEvent& event)
{
    ReadbackPixel pixel{};
    if (!isValidReadback(event))
    {
        return pixel;
    }

    return decodePixel(event, event.width / 2u, event.height / 2u);
}

template <typename ReadbackEvent, typename Predicate>
inline std::uint64_t countPixelsMatching(const ReadbackEvent& event, std::uint32_t xBegin,
                                         std::uint32_t xEnd,
                                         Predicate predicate)
{
    if (!isValidReadback(event) || xBegin >= xEnd || xEnd > event.width)
    {
        return 0u;
    }

    std::uint64_t count = 0u;
    for (std::uint32_t y = 0u; y < event.height; ++y)
    {
        for (std::uint32_t x = xBegin; x < xEnd; ++x)
        {
            if (predicate(decodePixel(event, x, y)))
            {
                ++count;
            }
        }
    }
    return count;
}

inline std::uint8_t encodeByte(float value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

template <typename ReadbackEvent>
inline bool writePpm(const std::string& path, const ReadbackEvent& event)
{
    if (!isValidReadback(event))
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    out << "P6\n" << event.width << " " << event.height << "\n255\n";

    std::vector<std::uint8_t> rgbRow(static_cast<std::size_t>(event.width) * 3u);
    for (std::uint32_t y = 0; y < event.height; ++y)
    {
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            const ReadbackPixel pixel = decodePixel(event, x, y);
            rgbRow[static_cast<std::size_t>(x) * 3u + 0u] = encodeByte(pixel.r);
            rgbRow[static_cast<std::size_t>(x) * 3u + 1u] = encodeByte(pixel.g);
            rgbRow[static_cast<std::size_t>(x) * 3u + 2u] = encodeByte(pixel.b);
        }

        out.write(reinterpret_cast<const char*>(rgbRow.data()),
                  static_cast<std::streamsize>(rgbRow.size()));
    }

    return out.good();
}

} // namespace cressim::neo::tests::helpers

#endif
