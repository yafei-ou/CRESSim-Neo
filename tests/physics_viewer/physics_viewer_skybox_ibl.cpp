#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/render_resource_manager.h"
#include "viewer/debug_viewer_app.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureDimension;
using cressim::neo::graphics::TexturePixelFormat;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;
constexpr std::uint32_t kIrradianceSize = 16u;
constexpr std::uint32_t kSpecularSize = 128u;
constexpr std::uint32_t kSpecularMipCount = 7u;
constexpr std::uint32_t kIrradianceSampleCount = 256u;
constexpr std::uint32_t kSpecularSampleCount = 128u;

struct FloatImage
{
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::vector<Diligent::float4> pixels;
};

GpuBackend parseBackend(const std::string &value)
{
    if (value == "null")
    {
        return GpuBackend::Null;
    }
    if (value == "vulkan")
    {
        return GpuBackend::Vulkan;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

void printUsage(const char *appName)
{
    CRESSIM_LOG_ERROR("Usage: ", appName, " [--backend vulkan|null] [--frames N]\n");
}

float srgbToLinear(float value)
{
    if (value <= 0.04045f)
    {
        return value / 12.92f;
    }
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

std::uint16_t encodeFloat16(float value)
{
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x007fffffu;

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
    encoded[0] = encodeFloat16(std::max(rgba.x, 0.0f));
    encoded[1] = encodeFloat16(std::max(rgba.y, 0.0f));
    encoded[2] = encodeFloat16(std::max(rgba.z, 0.0f));
    encoded[3] = encodeFloat16(std::max(rgba.w, 0.0f));
}

FloatImage loadFaceImage(const std::filesystem::path &path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc *bytes = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    if (bytes == nullptr || width <= 0 || height <= 0)
    {
        throw std::runtime_error("Failed to load cubemap face: " + path.string());
    }

    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> guard(bytes, &stbi_image_free);

    FloatImage image{};
    image.width = static_cast<std::uint32_t>(width);
    image.height = static_cast<std::uint32_t>(height);
    image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    for (std::size_t i = 0u; i < image.pixels.size(); ++i)
    {
        const float r = static_cast<float>(bytes[i * 4u + 0u]) / 255.0f;
        const float g = static_cast<float>(bytes[i * 4u + 1u]) / 255.0f;
        const float b = static_cast<float>(bytes[i * 4u + 2u]) / 255.0f;
        const float a = static_cast<float>(bytes[i * 4u + 3u]) / 255.0f;
        image.pixels[i] = {srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), a};
    }
    return image;
}

float clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

Diligent::float3 clamp01(const Diligent::float3 &value)
{
    return {clamp01(value.x), clamp01(value.y), clamp01(value.z)};
}

Diligent::float3 rgb(const Diligent::float4 &value)
{
    return {value.x, value.y, value.z};
}

float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

Diligent::float3 cubeDirection(std::uint32_t face, std::uint32_t x, std::uint32_t y,
                               std::uint32_t size)
{
    const float u = ((static_cast<float>(x) + 0.5f) / static_cast<float>(size)) * 2.0f - 1.0f;
    const float v = ((static_cast<float>(y) + 0.5f) / static_cast<float>(size)) * 2.0f - 1.0f;

    switch (face)
    {
    case 0u:
        return Diligent::normalize(Diligent::float3{1.0f, -v, -u});
    case 1u:
        return Diligent::normalize(Diligent::float3{-1.0f, -v, u});
    case 2u:
        return Diligent::normalize(Diligent::float3{u, 1.0f, v});
    case 3u:
        return Diligent::normalize(Diligent::float3{u, -1.0f, -v});
    case 4u:
        return Diligent::normalize(Diligent::float3{u, -v, 1.0f});
    default:
        return Diligent::normalize(Diligent::float3{-u, -v, -1.0f});
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
            outU = -dir.z / absDir.x;
            outV = -dir.y / absDir.x;
        }
        else
        {
            outFace = 1u;
            outU = dir.z / absDir.x;
            outV = -dir.y / absDir.x;
        }
    }
    else if (absDir.y >= absDir.x && absDir.y >= absDir.z)
    {
        if (dir.y >= 0.0f)
        {
            outFace = 2u;
            outU = dir.x / absDir.y;
            outV = dir.z / absDir.y;
        }
        else
        {
            outFace = 3u;
            outU = dir.x / absDir.y;
            outV = -dir.z / absDir.y;
        }
    }
    else
    {
        if (dir.z >= 0.0f)
        {
            outFace = 4u;
            outU = dir.x / absDir.z;
            outV = -dir.y / absDir.z;
        }
        else
        {
            outFace = 5u;
            outU = -dir.x / absDir.z;
            outV = -dir.y / absDir.z;
        }
    }

    outU = clamp01(outU * 0.5f + 0.5f);
    outV = clamp01(outV * 0.5f + 0.5f);
}

Diligent::float3 sampleBilinear(const FloatImage &image, float u, float v)
{
    const float x = clamp01(u) * static_cast<float>(image.width - 1u);
    const float y = clamp01(v) * static_cast<float>(image.height - 1u);
    const std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(x));
    const std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(y));
    const std::uint32_t x1 = std::min(x0 + 1u, image.width - 1u);
    const std::uint32_t y1 = std::min(y0 + 1u, image.height - 1u);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const Diligent::float3 c00 = rgb(image.pixels[static_cast<std::size_t>(y0) * image.width + x0]);
    const Diligent::float3 c10 = rgb(image.pixels[static_cast<std::size_t>(y0) * image.width + x1]);
    const Diligent::float3 c01 = rgb(image.pixels[static_cast<std::size_t>(y1) * image.width + x0]);
    const Diligent::float3 c11 = rgb(image.pixels[static_cast<std::size_t>(y1) * image.width + x1]);

    const Diligent::float3 top = c00 * (1.0f - tx) + c10 * tx;
    const Diligent::float3 bottom = c01 * (1.0f - tx) + c11 * tx;
    return top * (1.0f - ty) + bottom * ty;
}

Diligent::float3 sampleEnvironment(const std::array<FloatImage, 6u> &faces,
                                   const Diligent::float3 &dir)
{
    std::uint32_t face = 0u;
    float u = 0.5f;
    float v = 0.5f;
    directionToFaceUv(Diligent::normalize(dir), face, u, v);
    return sampleBilinear(faces[face], u, v);
}

void buildBasis(const Diligent::float3 &n, Diligent::float3 &tangent, Diligent::float3 &bitangent)
{
    const Diligent::float3 up =
        std::fabs(n.y) < 0.999f ? Diligent::float3{0.0f, 1.0f, 0.0f}
                                : Diligent::float3{1.0f, 0.0f, 0.0f};
    tangent = Diligent::normalize(Diligent::cross(up, n));
    bitangent = Diligent::normalize(Diligent::cross(n, tangent));
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
    const float a = std::max(roughness * roughness, 0.001f);
    const float phi = 2.0f * kPi * xi.x;
    const float cosTheta =
        std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));

    const Diligent::float3 hTangent{
        std::cos(phi) * sinTheta,
        std::sin(phi) * sinTheta,
        cosTheta,
    };

    Diligent::float3 tangent{};
    Diligent::float3 bitangent{};
    buildBasis(n, tangent, bitangent);
    return Diligent::normalize(tangent * hTangent.x + bitangent * hTangent.y + n * hTangent.z);
}

Diligent::float3 sampleCosineHemisphere(const Diligent::float2 &xi, const Diligent::float3 &n)
{
    const float phi = 2.0f * kPi * xi.x;
    const float cosTheta = std::sqrt(1.0f - xi.y);
    const float sinTheta = std::sqrt(xi.y);
    const Diligent::float3 local{
        std::cos(phi) * sinTheta,
        std::sin(phi) * sinTheta,
        cosTheta,
    };

    Diligent::float3 tangent{};
    Diligent::float3 bitangent{};
    buildBasis(n, tangent, bitangent);
    return Diligent::normalize(tangent * local.x + bitangent * local.y + n * local.z);
}

Diligent::float3 integrateIrradiance(const std::array<FloatImage, 6u> &faces,
                                     const Diligent::float3 &normal)
{
    Diligent::float3 sum{0.0f, 0.0f, 0.0f};
    for (std::uint32_t i = 0u; i < kIrradianceSampleCount; ++i)
    {
        const Diligent::float3 sampleDir =
            sampleCosineHemisphere(hammersley(i, kIrradianceSampleCount), normal);
        sum += sampleEnvironment(faces, sampleDir);
    }
    return sum * (kPi / static_cast<float>(kIrradianceSampleCount));
}

Diligent::float3 integratePrefilteredSpecular(const std::array<FloatImage, 6u> &faces,
                                              const Diligent::float3 &reflectionDir,
                                              float roughness)
{
    if (roughness <= 0.001f)
    {
        return sampleEnvironment(faces, reflectionDir);
    }

    const Diligent::float3 n = reflectionDir;
    const Diligent::float3 v = reflectionDir;
    Diligent::float3 sum{0.0f, 0.0f, 0.0f};
    float totalWeight = 0.0f;
    for (std::uint32_t i = 0u; i < kSpecularSampleCount; ++i)
    {
        const Diligent::float3 h = importanceSampleGgx(hammersley(i, kSpecularSampleCount), n,
                                                       roughness);
        const Diligent::float3 l = Diligent::normalize(2.0f * Diligent::dot(v, h) * h - v);
        const float nDotL = std::max(Diligent::dot(n, l), 0.0f);
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

TextureResourceDesc buildSpecularCubemap(const std::array<FloatImage, 6u> &faces)
{
    TextureResourceDesc desc{};
    desc.debugName = "ViewerIntegration.SkyboxIbl.Specular";
    desc.width = kSpecularSize;
    desc.height = kSpecularSize;
    desc.mipLevelCount = kSpecularMipCount;
    desc.dimension = TextureDimension::TextureCube;
    desc.pixelFormat = TexturePixelFormat::RGBA16F;
    desc.colorSpace = TextureColorSpace::Linear;
    desc.subresources.resize(static_cast<std::size_t>(desc.mipLevelCount) * 6u);

    for (std::uint32_t mip = 0u; mip < desc.mipLevelCount; ++mip)
    {
        const std::uint32_t mipSize = std::max(desc.width >> mip, 1u);
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
                        integratePrefilteredSpecular(faces, reflectionDir, roughness);
                    appendRgba16f(dst, {color.x, color.y, color.z, 1.0f});
                }
            }
        }
    }

    return desc;
}

TextureResourceDesc buildIrradianceCubemap(const std::array<FloatImage, 6u> &faces)
{
    TextureResourceDesc desc{};
    desc.debugName = "ViewerIntegration.SkyboxIbl.Irradiance";
    desc.width = kIrradianceSize;
    desc.height = kIrradianceSize;
    desc.mipLevelCount = 1u;
    desc.dimension = TextureDimension::TextureCube;
    desc.pixelFormat = TexturePixelFormat::RGBA16F;
    desc.colorSpace = TextureColorSpace::Linear;
    desc.subresources.resize(6u);

    for (std::uint32_t face = 0u; face < 6u; ++face)
    {
        auto &dst = desc.subresources[face].pixelData;
        dst.reserve(static_cast<std::size_t>(kIrradianceSize) * kIrradianceSize * 8u);
        for (std::uint32_t y = 0u; y < kIrradianceSize; ++y)
        {
            for (std::uint32_t x = 0u; x < kIrradianceSize; ++x)
            {
                const Diligent::float3 normal = cubeDirection(face, x, y, kIrradianceSize);
                const Diligent::float3 color = integrateIrradiance(faces, normal);
                appendRgba16f(dst, {color.x, color.y, color.z, 1.0f});
            }
        }
    }

    return desc;
}

EnvironmentIblDesc loadSkyboxIbl(cressim::neo::graphics::RenderResourceManager &resources)
{
    const std::filesystem::path skyboxDir =
        std::filesystem::path(CRESSIM_NEO_PROJECT_SOURCE_DIR) / "tests/physics_viewer/skybox";
    const std::array<const char *, 6u> faceNames = {
        "posx.jpg", "negx.jpg", "posy.jpg", "negy.jpg", "posz.jpg", "negz.jpg"};

    std::array<FloatImage, 6u> faces;
    for (std::uint32_t face = 0u; face < faceNames.size(); ++face)
    {
        faces[face] = loadFaceImage(skyboxDir / faceNames[face]);
    }

    const std::uint32_t faceWidth = faces[0u].width;
    const std::uint32_t faceHeight = faces[0u].height;
    for (std::uint32_t face = 1u; face < faces.size(); ++face)
    {
        if (faces[face].width != faceWidth || faces[face].height != faceHeight)
        {
            throw std::runtime_error("Skybox cubemap faces must all have matching dimensions.");
        }
    }

    EnvironmentIblDesc ibl{};
    ibl.irradianceCubemap = resources.registerTexture(buildIrradianceCubemap(faces));
    ibl.prefilteredSpecularCubemap = resources.registerTexture(buildSpecularCubemap(faces));
    ibl.intensity = 1.0f;
    return ibl;
}

MeshResourceDesc makePlaneMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "ViewerIntegration.SkyboxIbl.Plane";
    const float h = halfExtent;
    mesh.vertices = {
        {{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f},
        {{h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 1.0f, 0.0f},
        {{h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 1.0f, 1.0f},
        {{-h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f}};
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return mesh;
}

MeshResourceDesc makeSphereMesh(float radius, std::uint32_t slices, std::uint32_t stacks)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "ViewerIntegration.SkyboxIbl.Sphere";
    mesh.vertices.reserve((stacks + 1u) * (slices + 1u));
    mesh.indices.reserve(stacks * slices * 6u);

    for (std::uint32_t stack = 0u; stack <= stacks; ++stack)
    {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * kPi;
        const float y = std::cos(phi);
        const float ringRadius = std::sin(phi);

        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * kPi);
            const float x = ringRadius * std::cos(theta);
            const float z = ringRadius * std::sin(theta);
            const Diligent::float3 normal{x, y, z};
            mesh.vertices.push_back({normal * radius, normal, u, v});
        }
    }

    const std::uint32_t ring = slices + 1u;
    for (std::uint32_t stack = 0u; stack < stacks; ++stack)
    {
        for (std::uint32_t slice = 0u; slice < slices; ++slice)
        {
            const std::uint32_t i0 = stack * ring + slice;
            const std::uint32_t i1 = i0 + 1u;
            const std::uint32_t i2 = i0 + ring;
            const std::uint32_t i3 = i2 + 1u;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float metallic, float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic = metallic;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

} // namespace

int main(int argc, char **argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.rendererDesc.iblQualityTier = IblQualityTier::Full;
    config.sceneLayout.envCount = 1u;
    config.sceneLayout.maxRenderableObjectsPerEnv = 4u;
    config.sceneLayout.maxLightsPerEnv = 1u;
    config.sceneLayout.maxCamerasPerEnv = 1u;
    std::uint64_t numFrames = 0u;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--backend")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            config.gpuDeviceDesc.preferredBackend = parseBackend(argv[++i]);
            continue;
        }
        if (arg == "--frames")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            numFrames = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
            continue;
        }

        printUsage(argv[0]);
        return 2;
    }

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled = windowEnabled;
    viewerDesc.windowVisible = windowEnabled;
    viewerDesc.startFullscreenWindowed = false;
    viewerDesc.maxFrames = numFrames;
    viewerDesc.showStats = true;
    viewerDesc.vSync = true;
    viewerDesc.width = 1280;
    viewerDesc.height = 720;
    viewerDesc.windowTitle = "CRESSim Neo Physics Viewer Skybox IBL";

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    try
    {
        auto &world = runtime.getWorld();
        auto &resources = runtime.getResources();

        const auto cameraEntity = world.createEntity();
        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 1.2f, -4.0f};
        world.setTransform(cameraEntity, cameraTransform);
        world.setCamera(cameraEntity, CameraComponent{});

        const auto lightEntity = world.createEntity();
        DirectionalLightComponent light{};
        light.direction = {0.2f, -1.0f, 0.1f};
        light.color = {1.0f, 0.98f, 0.95f};
        light.intensity = 0.35f;
        world.setDirectionalLight(lightEntity, light);

        if (!world.setEnvironmentIbl(0u, loadSkyboxIbl(resources)))
        {
            throw std::runtime_error("Failed to assign skybox IBL to environment 0.");
        }

        const MeshHandle planeMesh = resources.registerMesh(makePlaneMesh(8.0f));
        const MeshHandle sphereMesh = resources.registerMesh(makeSphereMesh(1.0f, 48u, 24u));

        const MaterialHandle groundMaterial = registerMaterial(
            resources, "ViewerIntegration.SkyboxIbl.Ground", {0.35f, 0.36f, 0.38f}, 0.0f, 0.92f);

        MaterialResourceDesc shinySphereMaterialDesc{};
        shinySphereMaterialDesc.debugName = "ViewerIntegration.SkyboxIbl.ShinySphere";
        shinySphereMaterialDesc.baseColor = {0.98f, 0.98f, 0.98f};
        shinySphereMaterialDesc.metallic = 0.85f;
        shinySphereMaterialDesc.roughness = 0.05f;
        const MaterialHandle shinySphereMaterial = resources.registerMaterial(shinySphereMaterialDesc);

        const auto groundEntity = world.createEntity();
        TransformComponent groundTransform{};
        groundTransform.worldTransform.position = {0.0f, -1.05f, 0.0f};
        world.setTransform(groundEntity, groundTransform);
        MeshRendererComponent ground{};
        ground.mesh = planeMesh;
        ground.material = groundMaterial;
        ground.visible = true;
        world.setMeshRenderer(groundEntity, ground);

        const auto sphereEntity = world.createEntity();
        TransformComponent sphereTransform{};
        sphereTransform.worldTransform.position = {0.0f, 0.0f, 0.8f};
        world.setTransform(sphereEntity, sphereTransform);
        MeshRendererComponent sphere{};
        sphere.mesh = sphereMesh;
        sphere.material = shinySphereMaterial;
        sphere.visible = true;
        world.setMeshRenderer(sphereEntity, sphere);

        std::uint64_t beforeCalls = 0u;
        std::uint64_t afterCalls = 0u;
        DebugViewerCallbacks callbacks{};
        callbacks.beforeTick = [&](const FrameContext &, Runtime &) { ++beforeCalls; };
        callbacks.afterTick = [&](const FrameContext &, Runtime &) { ++afterCalls; };

        DebugViewerCameraBinding binding{};
        binding.cameraEntity = cameraEntity;
        const bool runOk = viewer.run(runtime, binding, callbacks);

        runtime.shutdown();
        viewer.shutdown();

        if (!runOk)
        {
            CRESSIM_LOG_ERROR("Viewer run failed.\n");
            return 1;
        }
        if (viewerDesc.maxFrames > 0 &&
            (beforeCalls != viewerDesc.maxFrames || afterCalls != viewerDesc.maxFrames))
        {
            CRESSIM_LOG_ERROR("Unexpected callback counts. before=", beforeCalls,
                              " after=", afterCalls, " expected=", viewerDesc.maxFrames, '\n');
            return 1;
        }
    }
    catch (const std::exception &ex)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Skybox IBL viewer setup failed: ", ex.what(), '\n');
        return 1;
    }

    CRESSIM_LOG_INFO("Physics viewer skybox IBL passed. Frames=", viewerDesc.maxFrames, '\n');
    return 0;
}
