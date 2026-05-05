#include "graphics/environment_ibl_baker.h"

#include "common/math_utils_runtime.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentTools/TextureLoader/interface/Image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cressim::neo::graphics
{

namespace
{
using cressim::neo::common::runtime_math::clamp01;
using cressim::neo::common::runtime_math::clampExtent;
using cressim::neo::common::runtime_math::kPi;
using cressim::neo::common::runtime_math::safeNormalize;

float srgbToLinear(float value)
{
    if (value <= 0.04045f)
    {
        return value / 12.92f;
    }
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

Diligent::float3 rgb(const Diligent::float4 &value)
{
    return {value.x, value.y, value.z};
}

std::uint16_t encodeFloat16(float value)
{
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::int32_t exponent    = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa   = bits & 0x007fffffu;

    if (exponent <= 0)
    {
        if (exponent < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }

        mantissa = (mantissa | 0x00800000u) >> static_cast<std::uint32_t>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
    }
    if (exponent >= 31)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10u) |
                                      ((mantissa + 0x00001000u) >> 13u));
}

void appendRgba16f(std::vector<std::uint8_t> &dst, const Diligent::float4 &rgba)
{
    const std::size_t offset = dst.size();
    dst.resize(offset + sizeof(std::uint16_t) * 4u);
    auto *encoded = reinterpret_cast<std::uint16_t *>(dst.data() + offset);
    encoded[0]    = encodeFloat16(std::max(rgba.x, 0.0f));
    encoded[1]    = encodeFloat16(std::max(rgba.y, 0.0f));
    encoded[2]    = encodeFloat16(std::max(rgba.z, 0.0f));
    encoded[3]    = encodeFloat16(std::max(rgba.w, 0.0f));
}

EnvironmentCubemapImage loadFaceImage(const std::filesystem::path &path)
{
    Diligent::RefCntAutoPtr<Diligent::Image> image;
    Diligent::CreateImageFromFile(path.string().c_str(), &image, nullptr);
    if (image == nullptr)
    {
        throw std::runtime_error("Failed to load cubemap face: " + path.string());
    }

    const Diligent::ImageDesc &desc = image->GetDesc();
    if (desc.ComponentType != Diligent::VT_UINT8 ||
        (desc.NumComponents != 3u && desc.NumComponents != 4u))
    {
        throw std::runtime_error("Unsupported cubemap face format: " + path.string());
    }
    if (desc.Width == 0u || desc.Height == 0u || image->GetData() == nullptr)
    {
        throw std::runtime_error("Cubemap face image is empty: " + path.string());
    }

    EnvironmentCubemapImage result{};
    result.width  = desc.Width;
    result.height = desc.Height;
    result.pixels.resize(static_cast<std::size_t>(result.width) * result.height);

    const auto *src = image->GetData()->GetConstDataPtr<std::uint8_t>();
    for (std::uint32_t y = 0u; y < result.height; ++y)
    {
        const auto *row = src + static_cast<std::size_t>(y) * desc.RowStride;
        for (std::uint32_t x = 0u; x < result.width; ++x)
        {
            const auto *pixel = row + static_cast<std::size_t>(x) * desc.NumComponents;
            const float r     = static_cast<float>(pixel[0]) / 255.0f;
            const float g     = static_cast<float>(pixel[1]) / 255.0f;
            const float b     = static_cast<float>(pixel[2]) / 255.0f;
            const float a = desc.NumComponents == 4u ? static_cast<float>(pixel[3]) / 255.0f : 1.0f;
            result.pixels[static_cast<std::size_t>(y) * result.width + x] = {
                srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), a};
        }
    }

    return result;
}

Diligent::float3 cubeDirection(std::uint32_t face, std::uint32_t x, std::uint32_t y,
                               std::uint32_t size)
{
    const float u = ((static_cast<float>(x) + 0.5f) / static_cast<float>(size)) * 2.0f - 1.0f;
    const float v = ((static_cast<float>(y) + 0.5f) / static_cast<float>(size)) * 2.0f - 1.0f;

    switch (face)
    {
    case 0u:
        return safeNormalize(Diligent::float3{1.0f, -v, -u});
    case 1u:
        return safeNormalize(Diligent::float3{-1.0f, -v, u});
    case 2u:
        return safeNormalize(Diligent::float3{u, 1.0f, v});
    case 3u:
        return safeNormalize(Diligent::float3{u, -1.0f, -v});
    case 4u:
        return safeNormalize(Diligent::float3{u, -v, 1.0f});
    default:
        return safeNormalize(Diligent::float3{-u, -v, -1.0f});
    }
}

void directionToFaceUv(const Diligent::float3 &dir, std::uint32_t &outFace, float &outU,
                       float &outV)
{
    const Diligent::float3 absDir{std::fabs(dir.x), std::fabs(dir.y), std::fabs(dir.z)};
    if (absDir.x >= absDir.y && absDir.x >= absDir.z)
    {
        if (dir.x >= 0.0f)
        {
            outFace = 0u;
            outU    = -dir.z / absDir.x;
            outV    = -dir.y / absDir.x;
        }
        else
        {
            outFace = 1u;
            outU    = dir.z / absDir.x;
            outV    = -dir.y / absDir.x;
        }
    }
    else if (absDir.y >= absDir.x && absDir.y >= absDir.z)
    {
        if (dir.y >= 0.0f)
        {
            outFace = 2u;
            outU    = dir.x / absDir.y;
            outV    = dir.z / absDir.y;
        }
        else
        {
            outFace = 3u;
            outU    = dir.x / absDir.y;
            outV    = -dir.z / absDir.y;
        }
    }
    else
    {
        if (dir.z >= 0.0f)
        {
            outFace = 4u;
            outU    = dir.x / absDir.z;
            outV    = -dir.y / absDir.z;
        }
        else
        {
            outFace = 5u;
            outU    = -dir.x / absDir.z;
            outV    = -dir.y / absDir.z;
        }
    }

    outU = clamp01(outU * 0.5f + 0.5f);
    outV = clamp01(outV * 0.5f + 0.5f);
}

Diligent::float3 sampleBilinear(const EnvironmentCubemapImage &image, float u, float v)
{
    const float x          = clamp01(u) * static_cast<float>(image.width - 1u);
    const float y          = clamp01(v) * static_cast<float>(image.height - 1u);
    const std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(x));
    const std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(y));
    const std::uint32_t x1 = std::min(x0 + 1u, image.width - 1u);
    const std::uint32_t y1 = std::min(y0 + 1u, image.height - 1u);
    const float tx         = x - static_cast<float>(x0);
    const float ty         = y - static_cast<float>(y0);

    const Diligent::float3 c00 = rgb(image.pixels[static_cast<std::size_t>(y0) * image.width + x0]);
    const Diligent::float3 c10 = rgb(image.pixels[static_cast<std::size_t>(y0) * image.width + x1]);
    const Diligent::float3 c01 = rgb(image.pixels[static_cast<std::size_t>(y1) * image.width + x0]);
    const Diligent::float3 c11 = rgb(image.pixels[static_cast<std::size_t>(y1) * image.width + x1]);

    const Diligent::float3 top    = c00 * (1.0f - tx) + c10 * tx;
    const Diligent::float3 bottom = c01 * (1.0f - tx) + c11 * tx;
    return top * (1.0f - ty) + bottom * ty;
}

Diligent::float3 sampleEnvironment(const std::array<EnvironmentCubemapImage, 6u> &faces,
                                   const Diligent::float3 &dir)
{
    std::uint32_t face = 0u;
    float u            = 0.5f;
    float v            = 0.5f;
    directionToFaceUv(safeNormalize(dir), face, u, v);
    return sampleBilinear(faces[face], u, v);
}

void buildBasis(const Diligent::float3 &n, Diligent::float3 &tangent, Diligent::float3 &bitangent)
{
    const Diligent::float3 up = std::fabs(n.y) < 0.999f ? Diligent::float3{0.0f, 1.0f, 0.0f}
                                                        : Diligent::float3{1.0f, 0.0f, 0.0f};
    tangent   = safeNormalize(Diligent::cross(up, n), Diligent::float3{1.0f, 0.0f, 0.0f});
    bitangent = safeNormalize(Diligent::cross(n, tangent), Diligent::float3{0.0f, 1.0f, 0.0f});
}

float radicalInverseVdc(std::uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

Diligent::float2 hammersley(std::uint32_t i, std::uint32_t count)
{
    return {static_cast<float>(i) / static_cast<float>(count), radicalInverseVdc(i)};
}

Diligent::float3 importanceSampleGgx(const Diligent::float2 &xi, const Diligent::float3 &n,
                                     float roughness)
{
    const float a        = std::max(roughness * roughness, 0.001f);
    const float phi      = 2.0f * kPi * xi.x;
    const float cosTheta = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));

    const Diligent::float3 hTangent{std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};

    Diligent::float3 tangent{};
    Diligent::float3 bitangent{};
    buildBasis(n, tangent, bitangent);
    return safeNormalize(tangent * hTangent.x + bitangent * hTangent.y + n * hTangent.z, n);
}

Diligent::float3 sampleCosineHemisphere(const Diligent::float2 &xi, const Diligent::float3 &n)
{
    const float phi      = 2.0f * kPi * xi.x;
    const float cosTheta = std::sqrt(1.0f - xi.y);
    const float sinTheta = std::sqrt(xi.y);
    const Diligent::float3 local{std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};

    Diligent::float3 tangent{};
    Diligent::float3 bitangent{};
    buildBasis(n, tangent, bitangent);
    return safeNormalize(tangent * local.x + bitangent * local.y + n * local.z, n);
}

Diligent::float3 integrateIrradiance(const std::array<EnvironmentCubemapImage, 6u> &faces,
                                     const Diligent::float3 &normal, std::uint32_t sampleCount)
{
    Diligent::float3 sum{0.0f, 0.0f, 0.0f};
    for (std::uint32_t i = 0u; i < sampleCount; ++i)
    {
        const Diligent::float3 sampleDir =
            sampleCosineHemisphere(hammersley(i, sampleCount), normal);
        sum += sampleEnvironment(faces, sampleDir);
    }
    return sum * (kPi / static_cast<float>(sampleCount));
}

Diligent::float3 integratePrefilteredSpecular(const std::array<EnvironmentCubemapImage, 6u> &faces,
                                              const Diligent::float3 &reflectionDir,
                                              float roughness, std::uint32_t sampleCount)
{
    if (roughness <= 0.001f)
    {
        return sampleEnvironment(faces, reflectionDir);
    }

    const Diligent::float3 n = reflectionDir;
    const Diligent::float3 v = reflectionDir;
    Diligent::float3 sum{0.0f, 0.0f, 0.0f};
    float totalWeight = 0.0f;
    for (std::uint32_t i = 0u; i < sampleCount; ++i)
    {
        const Diligent::float3 h = importanceSampleGgx(hammersley(i, sampleCount), n, roughness);
        const Diligent::float3 l = safeNormalize(2.0f * Diligent::dot(v, h) * h - v, reflectionDir);
        const float nDotL        = std::max(Diligent::dot(n, l), 0.0f);
        if (nDotL <= 0.0f)
        {
            continue;
        }

        sum += sampleEnvironment(faces, l) * nDotL;
        totalWeight += nDotL;
    }

    if (totalWeight <= 1.0e-5f)
    {
        return sampleEnvironment(faces, reflectionDir);
    }
    return sum / totalWeight;
}

TextureResourceDesc buildIrradianceCubemap(const std::array<EnvironmentCubemapImage, 6u> &faces,
                                           const EnvironmentIblBakeOptions &options)
{
    TextureResourceDesc desc{};
    desc.debugName     = "EnvironmentIblBaker.Irradiance";
    desc.width         = clampExtent(options.irradianceSize);
    desc.height        = clampExtent(options.irradianceSize);
    desc.mipLevelCount = 1u;
    desc.dimension     = TextureDimension::TextureCube;
    desc.pixelFormat   = TexturePixelFormat::RGBA16F;
    desc.colorSpace    = TextureColorSpace::Linear;
    desc.subresources.resize(6u);

    const std::uint32_t sampleCount = clampExtent(options.irradianceSampleCount);
    for (std::uint32_t face = 0u; face < 6u; ++face)
    {
        auto &dst = desc.subresources[face].pixelData;
        dst.reserve(static_cast<std::size_t>(desc.width) * desc.height * 8u);
        for (std::uint32_t y = 0u; y < desc.height; ++y)
        {
            for (std::uint32_t x = 0u; x < desc.width; ++x)
            {
                const Diligent::float3 normal = cubeDirection(face, x, y, desc.width);
                const Diligent::float3 color  = integrateIrradiance(faces, normal, sampleCount);
                appendRgba16f(dst, {color.x, color.y, color.z, 1.0f});
            }
        }
    }

    return desc;
}

TextureResourceDesc buildSpecularCubemap(const std::array<EnvironmentCubemapImage, 6u> &faces,
                                         const EnvironmentIblBakeOptions &options)
{
    TextureResourceDesc desc{};
    desc.debugName     = "EnvironmentIblBaker.Specular";
    desc.width         = clampExtent(options.specularSize);
    desc.height        = clampExtent(options.specularSize);
    desc.mipLevelCount = clampExtent(options.specularMipCount);
    desc.dimension     = TextureDimension::TextureCube;
    desc.pixelFormat   = TexturePixelFormat::RGBA16F;
    desc.colorSpace    = TextureColorSpace::Linear;
    desc.subresources.resize(static_cast<std::size_t>(desc.mipLevelCount) * 6u);

    const std::uint32_t sampleCount = clampExtent(options.specularSampleCount);
    for (std::uint32_t mip = 0u; mip < desc.mipLevelCount; ++mip)
    {
        const std::uint32_t mipSize = clampExtent(desc.width >> mip);
        const float roughness =
            static_cast<float>(mip) / static_cast<float>(std::max(desc.mipLevelCount - 1u, 1u));
        for (std::uint32_t face = 0u; face < 6u; ++face)
        {
            auto &dst = desc.subresources[static_cast<std::size_t>(mip) * 6u + face].pixelData;
            dst.reserve(static_cast<std::size_t>(mipSize) * mipSize * 8u);
            for (std::uint32_t y = 0u; y < mipSize; ++y)
            {
                for (std::uint32_t x = 0u; x < mipSize; ++x)
                {
                    const Diligent::float3 reflectionDir = cubeDirection(face, x, y, mipSize);
                    const Diligent::float3 color =
                        integratePrefilteredSpecular(faces, reflectionDir, roughness, sampleCount);
                    appendRgba16f(dst, {color.x, color.y, color.z, 1.0f});
                }
            }
        }
    }

    return desc;
}

TextureResourceDesc buildBackgroundCubemap(const std::array<EnvironmentCubemapImage, 6u> &faces)
{
    TextureResourceDesc desc{};
    desc.debugName     = "EnvironmentIblBaker.Background";
    desc.width         = faces[0u].width;
    desc.height        = faces[0u].height;
    desc.mipLevelCount = 1u;
    desc.dimension     = TextureDimension::TextureCube;
    desc.pixelFormat   = TexturePixelFormat::RGBA16F;
    desc.colorSpace    = TextureColorSpace::Linear;
    desc.subresources.resize(6u);

    for (std::uint32_t face = 0u; face < 6u; ++face)
    {
        auto &dst = desc.subresources[face].pixelData;
        dst.reserve(static_cast<std::size_t>(faces[face].width) * faces[face].height *
                    sizeof(std::uint16_t) * 4u);
        for (const Diligent::float4 &rgba : faces[face].pixels)
        {
            appendRgba16f(dst, rgba);
        }
    }

    return desc;
}

} // namespace

EnvironmentIblDesc createEnvironmentIblFromCubemapImages(
    RenderResourceManager &resources, const std::array<EnvironmentCubemapImage, 6u> &faces,
    const EnvironmentIblBakeOptions &options)
{
    const std::uint32_t faceWidth  = faces[0u].width;
    const std::uint32_t faceHeight = faces[0u].height;
    for (std::uint32_t face = 1u; face < faces.size(); ++face)
    {
        if (faces[face].width != faceWidth || faces[face].height != faceHeight)
        {
            throw std::runtime_error("Skybox cubemap faces must all have matching dimensions.");
        }
    }

    EnvironmentIblDesc ibl{};
    ibl.backgroundCubemap = resources.registerTexture(buildBackgroundCubemap(faces));
    ibl.irradianceCubemap = resources.registerTexture(buildIrradianceCubemap(faces, options));
    ibl.prefilteredSpecularCubemap =
        resources.registerTexture(buildSpecularCubemap(faces, options));
    ibl.intensity           = std::max(options.intensity, 0.0f);
    ibl.backgroundIntensity = 1.0f;
    return ibl;
}

EnvironmentIblDesc createEnvironmentIblFromCubemapFiles(
    RenderResourceManager &resources, const std::array<std::filesystem::path, 6u> &facePaths,
    const EnvironmentIblBakeOptions &options)
{
    std::array<EnvironmentCubemapImage, 6u> faces;
    for (std::uint32_t face = 0u; face < facePaths.size(); ++face)
    {
        faces[face] = loadFaceImage(facePaths[face]);
    }
    return createEnvironmentIblFromCubemapImages(resources, faces, options);
}

} // namespace cressim::neo::graphics
