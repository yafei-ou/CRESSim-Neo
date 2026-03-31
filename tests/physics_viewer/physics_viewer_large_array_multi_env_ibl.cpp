#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/render_resource_manager.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureDimension;
using cressim::neo::graphics::TexturePixelFormat;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;
constexpr int kGridWidth = 8;
constexpr int kGridDepth = 8;
constexpr int kLayers = 4;
constexpr float kSpacing = 1.45f;
constexpr float kBaseHeight = 1.6f;
constexpr float kLayerHeight = 2.05f;
constexpr float kEnvWorldSpacing = 74.0f;
constexpr std::uint32_t kDynamicBodiesPerEnv =
    static_cast<std::uint32_t>(kGridWidth * kGridDepth * kLayers);
constexpr std::uint32_t kObjectsPerEnvBudget = kDynamicBodiesPerEnv + 32u;
constexpr std::uint32_t kIrradianceSize = 16u;
constexpr std::uint32_t kSpecularSize = 32u;
constexpr std::uint32_t kSpecularMipCount = 6u;

struct EnvMaterialSet
{
    MaterialHandle mirrorSphere;
    MaterialHandle polishedBox;
    MaterialHandle roughMetalCapsule;
    MaterialHandle dielectricSphere;
    MaterialHandle ground;
};

struct EnvIblPalette
{
    Diligent::float3 zenith{};
    Diligent::float3 horizon{};
    Diligent::float3 ground{};
    Diligent::float3 accent{};
    Diligent::float3 sunDirection{0.0f, 1.0f, 0.0f};
    Diligent::float3 sunColor{1.0f, 1.0f, 1.0f};
    Diligent::float3 averageRadiance{0.3f, 0.3f, 0.3f};
    float sunIntensity = 6.0f;
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
    CRESSIM_LOG_ERROR("Usage: ", appName,
                      " [--backend vulkan|null] [--frames N] [--envs N]\n");
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

float saturate(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

Diligent::float3 lerp(const Diligent::float3 &a, const Diligent::float3 &b, float t)
{
    return {lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t)};
}

Diligent::float3 multiply(const Diligent::float3 &a, const Diligent::float3 &b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

void appendRgba16f(std::vector<std::uint8_t> &dst, const Diligent::float3 &rgb, float alpha = 1.0f)
{
    const std::size_t offset = dst.size();
    dst.resize(offset + sizeof(std::uint16_t) * 4u);
    auto *encoded = reinterpret_cast<std::uint16_t *>(dst.data() + offset);
    encoded[0] = encodeFloat16(std::max(rgb.x, 0.0f));
    encoded[1] = encodeFloat16(std::max(rgb.y, 0.0f));
    encoded[2] = encodeFloat16(std::max(rgb.z, 0.0f));
    encoded[3] = encodeFloat16(std::max(alpha, 0.0f));
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

EnvIblPalette paletteForEnv(std::uint32_t envIndex)
{
    switch (envIndex % 4u)
    {
    case 0u:
        return {
            {0.20f, 0.38f, 0.92f},
            {0.95f, 0.58f, 0.22f},
            {0.12f, 0.09f, 0.07f},
            {0.25f, 0.72f, 1.15f},
            Diligent::normalize(Diligent::float3{0.62f, 0.74f, -0.28f}),
            {1.0f, 0.92f, 0.78f},
            {0.34f, 0.26f, 0.20f},
            8.5f};
    case 1u:
        return {
            {0.05f, 0.22f, 0.48f},
            {0.18f, 0.72f, 0.92f},
            {0.02f, 0.05f, 0.08f},
            {0.95f, 0.30f, 0.82f},
            Diligent::normalize(Diligent::float3{-0.38f, 0.86f, 0.34f}),
            {0.86f, 0.94f, 1.0f},
            {0.14f, 0.28f, 0.36f},
            7.0f};
    case 2u:
        return {
            {0.62f, 0.70f, 0.84f},
            {0.92f, 0.82f, 0.70f},
            {0.20f, 0.18f, 0.15f},
            {1.10f, 0.65f, 0.24f},
            Diligent::normalize(Diligent::float3{0.24f, 0.92f, 0.30f}),
            {1.0f, 0.88f, 0.70f},
            {0.46f, 0.40f, 0.34f},
            6.5f};
    default:
        return {
            {0.08f, 0.06f, 0.16f},
            {0.82f, 0.18f, 0.24f},
            {0.03f, 0.02f, 0.04f},
            {0.34f, 0.95f, 0.62f},
            Diligent::normalize(Diligent::float3{-0.55f, 0.78f, -0.30f}),
            {1.0f, 0.78f, 0.72f},
            {0.24f, 0.10f, 0.12f},
            9.0f};
    }
}

Diligent::float3 sampleEnvironmentRadiance(const EnvIblPalette &palette,
                                           const Diligent::float3 &dir, float roughness,
                                           bool includeSun)
{
    const float up = saturate(dir.y * 0.5f + 0.5f);
    const float skyMix = std::pow(up, 0.55f);
    const float groundMix = std::pow(1.0f - up, 1.45f);
    const float horizonBand = std::pow(1.0f - std::fabs(dir.y), 2.25f);
    const float azimuth =
        0.5f + 0.5f * std::sin(std::atan2(dir.z, dir.x) * 3.0f + roughness * 1.7f);

    Diligent::float3 color = lerp(palette.horizon, palette.zenith, skyMix);
    color = lerp(color, palette.ground, groundMix * 0.85f);
    color += palette.accent * (horizonBand * (0.20f + 0.40f * azimuth) * (1.0f - 0.45f * roughness));

    if (includeSun)
    {
        const float sunAlignment = saturate(Diligent::dot(dir, palette.sunDirection));
        const float exponent = lerp(96.0f, 6.0f, roughness);
        const float sunGlow = std::pow(sunAlignment, exponent) * palette.sunIntensity;
        color += palette.sunColor * sunGlow;
    }

    return lerp(color, palette.averageRadiance, roughness * roughness * 0.92f);
}

TextureResourceDesc makeIrradianceCubeDesc(std::uint32_t envIndex)
{
    TextureResourceDesc desc{};
    desc.debugName = "ViewerIntegration.Ibl.Irradiance." + std::to_string(envIndex);
    desc.width = kIrradianceSize;
    desc.height = kIrradianceSize;
    desc.mipLevelCount = 1u;
    desc.dimension = TextureDimension::TextureCube;
    desc.pixelFormat = TexturePixelFormat::RGBA16F;
    desc.colorSpace = TextureColorSpace::Linear;
    desc.subresources.resize(6u);

    const EnvIblPalette palette = paletteForEnv(envIndex);
    for (std::uint32_t face = 0u; face < 6u; ++face)
    {
        auto &pixels = desc.subresources[face].pixelData;
        pixels.reserve(static_cast<std::size_t>(kIrradianceSize) * kIrradianceSize * 8u);
        for (std::uint32_t y = 0u; y < kIrradianceSize; ++y)
        {
            for (std::uint32_t x = 0u; x < kIrradianceSize; ++x)
            {
                const Diligent::float3 dir = cubeDirection(face, x, y, kIrradianceSize);
                const Diligent::float3 rgb = sampleEnvironmentRadiance(palette, dir, 1.0f, false);
                appendRgba16f(pixels, rgb);
            }
        }
    }

    return desc;
}

TextureResourceDesc makePrefilteredSpecularCubeDesc(std::uint32_t envIndex)
{
    TextureResourceDesc desc{};
    desc.debugName = "ViewerIntegration.Ibl.Specular." + std::to_string(envIndex);
    desc.width = kSpecularSize;
    desc.height = kSpecularSize;
    desc.mipLevelCount = kSpecularMipCount;
    desc.dimension = TextureDimension::TextureCube;
    desc.pixelFormat = TexturePixelFormat::RGBA16F;
    desc.colorSpace = TextureColorSpace::Linear;
    desc.subresources.resize(static_cast<std::size_t>(kSpecularMipCount) * 6u);

    const EnvIblPalette palette = paletteForEnv(envIndex);
    for (std::uint32_t mip = 0u; mip < kSpecularMipCount; ++mip)
    {
        const std::uint32_t mipSize = std::max(kSpecularSize >> mip, 1u);
        const float roughness = static_cast<float>(mip) /
                                static_cast<float>(std::max(kSpecularMipCount - 1u, 1u));
        for (std::uint32_t face = 0u; face < 6u; ++face)
        {
            auto &pixels = desc.subresources[mip * 6u + face].pixelData;
            pixels.reserve(static_cast<std::size_t>(mipSize) * mipSize * 8u);
            for (std::uint32_t y = 0u; y < mipSize; ++y)
            {
                for (std::uint32_t x = 0u; x < mipSize; ++x)
                {
                    const Diligent::float3 dir = cubeDirection(face, x, y, mipSize);
                    const Diligent::float3 rgb =
                        sampleEnvironmentRadiance(palette, dir, roughness, true);
                    appendRgba16f(pixels, rgb);
                }
            }
        }
    }

    return desc;
}

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "ViewerIntegration.Ibl.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3 &normal, const Diligent::float3 &v0,
                             const Diligent::float3 &v1, const Diligent::float3 &v2,
                             const Diligent::float3 &v3)
    {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 1u);
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 3u);
        mesh.indices.push_back(base + 2u);
    };

    const float h = halfExtent;
    addFace({0.0f, 0.0f, 1.0f}, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h});
    addFace({0.0f, 0.0f, -1.0f}, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h});
    addFace({-1.0f, 0.0f, 0.0f}, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h});
    addFace({1.0f, 0.0f, 0.0f}, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h});
    addFace({0.0f, 1.0f, 0.0f}, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h});
    addFace({0.0f, -1.0f, 0.0f}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h});
    return mesh;
}

MeshResourceDesc makePlaneMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "ViewerIntegration.Ibl.PlaneMesh";
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
    mesh.debugName = "ViewerIntegration.Ibl.SphereMesh";
    mesh.vertices.reserve((stacks + 1u) * (slices + 1u));
    mesh.indices.reserve(stacks * slices * 6u);

    for (std::uint32_t stack = 0u; stack <= stacks; ++stack)
    {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * kPi;
        const float sy = std::cos(phi);
        const float ringRadius = std::sin(phi);

        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * kPi);
            const float sx = ringRadius * std::cos(theta);
            const float sz = ringRadius * std::sin(theta);
            const Diligent::float3 normal{sx, sy, sz};
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

MeshResourceDesc makeCapsuleMesh(float radius, float halfHeight, std::uint32_t slices,
                                 std::uint32_t hemisphereRings, std::uint32_t bodyRings)
{
    struct Ring
    {
        float y = 0.0f;
        float r = 0.0f;
    };

    MeshResourceDesc mesh{};
    mesh.debugName = "ViewerIntegration.Ibl.CapsuleMesh";
    std::vector<Ring> rings;
    rings.reserve(2u * hemisphereRings + bodyRings + 2u);

    rings.push_back({halfHeight + radius, 0.0f});
    for (std::uint32_t i = 1u; i <= hemisphereRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(hemisphereRings);
        const float angle = t * (kPi * 0.5f);
        rings.push_back({halfHeight + radius * std::cos(angle), radius * std::sin(angle)});
    }

    for (std::uint32_t i = 1u; i <= bodyRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(bodyRings + 1u);
        rings.push_back({halfHeight * (1.0f - 2.0f * t), radius});
    }

    for (std::uint32_t i = 1u; i <= hemisphereRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(hemisphereRings);
        const float angle = t * (kPi * 0.5f);
        rings.push_back({-halfHeight - radius * std::sin(angle), radius * std::cos(angle)});
    }

    mesh.vertices.reserve(static_cast<std::size_t>(rings.size()) * (slices + 1u));
    mesh.indices.reserve((static_cast<std::uint32_t>(rings.size()) - 1u) * slices * 6u);

    for (std::size_t ringIndex = 0u; ringIndex < rings.size(); ++ringIndex)
    {
        const float y = rings[ringIndex].y;
        const float rr = rings[ringIndex].r;
        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * kPi);
            const float x = rr * std::cos(theta);
            const float z = rr * std::sin(theta);

            Diligent::float3 normal{};
            if (y > halfHeight)
            {
                normal = Diligent::normalize(Diligent::float3{x, y - halfHeight, z});
            }
            else if (y < -halfHeight)
            {
                normal = Diligent::normalize(Diligent::float3{x, y + halfHeight, z});
            }
            else
            {
                normal = rr > 0.0f ? Diligent::normalize(Diligent::float3{x, 0.0f, z})
                                   : Diligent::float3{0.0f, 1.0f, 0.0f};
            }

            const float v = static_cast<float>(ringIndex) /
                            static_cast<float>(std::max<std::size_t>(1u, rings.size() - 1u));
            mesh.vertices.push_back({{x, y, z}, normal, u, v});
        }
    }

    const std::uint32_t ringStride = slices + 1u;
    for (std::uint32_t ringIndex = 0u;
         ringIndex + 1u < static_cast<std::uint32_t>(rings.size()); ++ringIndex)
    {
        for (std::uint32_t slice = 0u; slice < slices; ++slice)
        {
            const std::uint32_t i0 = ringIndex * ringStride + slice;
            const std::uint32_t i1 = i0 + 1u;
            const std::uint32_t i2 = i0 + ringStride;
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

Diligent::float3 computeBoxInverseInertia(const Diligent::float3 &halfExtents, float inverseMass)
{
    if (inverseMass <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float ix = mass * (halfExtents.y * halfExtents.y + halfExtents.z * halfExtents.z) / 3.0f;
    const float iy = mass * (halfExtents.x * halfExtents.x + halfExtents.z * halfExtents.z) / 3.0f;
    const float iz = mass * (halfExtents.x * halfExtents.x + halfExtents.y * halfExtents.y) / 3.0f;

    return {ix > 0.0f ? 1.0f / ix : 0.0f, iy > 0.0f ? 1.0f / iy : 0.0f,
            iz > 0.0f ? 1.0f / iz : 0.0f};
}

Diligent::float3 computeSphereInverseInertia(float radius, float inverseMass)
{
    if (inverseMass <= 0.0f || radius <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float inertia = 0.4f * mass * radius * radius;
    const float inverseInertia = inertia > 0.0f ? 1.0f / inertia : 0.0f;
    return {inverseInertia, inverseInertia, inverseInertia};
}

Diligent::float4 colliderParamsForShape(ColliderShapeType shape)
{
    switch (shape)
    {
    case ColliderShapeType::Sphere:
        return {0.45f, 0.0f, 0.0f, 0.0f};
    case ColliderShapeType::Box:
        return {0.45f, 0.45f, 0.45f, 0.0f};
    case ColliderShapeType::Capsule:
        return {0.28f, 0.52f, 0.0f, 0.0f};
    }

    return {0.45f, 0.45f, 0.45f, 0.0f};
}

Diligent::float3 inverseInertiaForShape(ColliderShapeType shape,
                                        const Diligent::float4 &colliderParams,
                                        float inverseMass)
{
    switch (shape)
    {
    case ColliderShapeType::Sphere:
        return computeSphereInverseInertia(colliderParams.x, inverseMass);
    case ColliderShapeType::Box:
        return computeBoxInverseInertia({colliderParams.x, colliderParams.y, colliderParams.z},
                                        inverseMass);
    case ColliderShapeType::Capsule:
        return computeBoxInverseInertia(
            {colliderParams.x, colliderParams.y + colliderParams.x, colliderParams.x},
            inverseMass);
    }

    return {0.0f, 0.0f, 0.0f};
}

Diligent::float3 envWorldOrigin(std::uint32_t envIndex, std::uint32_t envCount)
{
    const std::uint32_t cols = std::max(
        1u, static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<float>(envCount)))));
    const std::uint32_t rows = std::max(1u, (envCount + cols - 1u) / cols);
    const std::uint32_t col = envIndex % cols;
    const std::uint32_t row = envIndex / cols;

    const float xCenter = (static_cast<float>(cols) - 1.0f) * 0.5f;
    const float zCenter = (static_cast<float>(rows) - 1.0f) * 0.5f;
    return {(static_cast<float>(col) - xCenter) * kEnvWorldSpacing, 0.0f,
            (static_cast<float>(row) - zCenter) * kEnvWorldSpacing};
}

void createHeroObjects(cressim::neo::engine::World &world, std::uint32_t envIndex,
                       const Diligent::float3 &envOrigin, MeshHandle sphereMesh,
                       MeshHandle cubeMesh, MeshHandle capsuleMesh,
                       const EnvMaterialSet &materials)
{
    const std::array<Diligent::float3, 4> heroPositions = {
        envOrigin + Diligent::float3{-5.5f, 1.0f, -7.0f},
        envOrigin + Diligent::float3{-1.8f, 1.0f, -7.0f},
        envOrigin + Diligent::float3{1.8f, 1.0f, -7.0f},
        envOrigin + Diligent::float3{5.5f, 1.0f, -7.0f},
    };
    const std::array<MeshHandle, 4> heroMeshes = {sphereMesh, sphereMesh, cubeMesh, capsuleMesh};
    const std::array<MaterialHandle, 4> heroMaterials = {
        materials.mirrorSphere,
        materials.dielectricSphere,
        materials.polishedBox,
        materials.roughMetalCapsule,
    };

    for (std::size_t i = 0u; i < heroPositions.size(); ++i)
    {
        const auto entity = world.createEntity(envIndex);
        TransformComponent transform{};
        transform.worldTransform.position = heroPositions[i];
        if (i == 2u)
        {
            transform.worldTransform.scale = {1.2f, 1.2f, 1.2f};
        }
        world.setTransform(entity, transform);

        MeshRendererComponent renderer{};
        renderer.mesh = heroMeshes[i];
        renderer.material = heroMaterials[i];
        renderer.visible = true;
        world.setMeshRenderer(entity, renderer);
    }
}

void authorEnvironment(cressim::neo::engine::World &world, std::uint32_t envIndex,
                       std::uint32_t envCount, MeshHandle cubeMesh, MeshHandle planeMesh,
                       MeshHandle sphereMesh, MeshHandle capsuleMesh,
                       const EnvMaterialSet &materials,
                       cressim::neo::common::EntityId &outCameraEntity)
{
    const Diligent::float3 envOrigin = envWorldOrigin(envIndex, envCount);
    const float envPhase = static_cast<float>(envIndex) * 0.63f;
    const float envVelocityBiasX = std::cos(envPhase) * 0.05f;
    const float envVelocityBiasZ = std::sin(envPhase) * 0.05f;
    const float envAngularBias = 0.12f + 0.04f * static_cast<float>(envIndex % 5u);

    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = envOrigin + Diligent::float3{0.0f, 6.0f, -23.0f};
    world.setTransform(outCameraEntity, cameraTransform);

    CameraComponent camera{};
    camera.verticalFovDegrees = 52.0f;
    camera.viewport = {};
    camera.clearColor = true;
    camera.clearDepth = true;
    camera.renderOrder = static_cast<int>(envIndex);
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    const float lightYaw = 0.45f + static_cast<float>(envIndex) * 0.55f;
    DirectionalLightComponent light{};
    light.direction =
        Diligent::normalize(Diligent::float3{std::cos(lightYaw), -0.85f, std::sin(lightYaw)});
    light.color = {1.0f, 0.97f, 0.92f};
    light.intensity = 2.4f;
    light.castsShadows = (envIndex % 2u) == 0u;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = envOrigin + Diligent::float3{0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent ground{};
    ground.mesh = planeMesh;
    ground.material = materials.ground;
    ground.visible = true;
    world.setMeshRenderer(groundEntity, ground);
    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = cressim::neo::physics::RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    cressim::neo::engine::ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {28.0f, 0.05f, 28.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    createHeroObjects(world, envIndex, envOrigin, sphereMesh, cubeMesh, capsuleMesh, materials);

    const auto meshForShape = [&](ColliderShapeType shape)
    {
        switch (shape)
        {
        case ColliderShapeType::Sphere:
            return sphereMesh;
        case ColliderShapeType::Box:
            return cubeMesh;
        case ColliderShapeType::Capsule:
            return capsuleMesh;
        }
        return cubeMesh;
    };

    const auto materialForShape = [&](ColliderShapeType shape)
    {
        switch (shape)
        {
        case ColliderShapeType::Sphere:
            return materials.mirrorSphere;
        case ColliderShapeType::Box:
            return materials.polishedBox;
        case ColliderShapeType::Capsule:
            return materials.roughMetalCapsule;
        }
        return materials.polishedBox;
    };

    const float xOrigin = -0.5f * static_cast<float>(kGridWidth - 1) * kSpacing;
    const float zOrigin = -0.5f * static_cast<float>(kGridDepth - 1) * kSpacing + 7.0f;

    for (int layer = 0; layer < kLayers; ++layer)
    {
        for (int z = 0; z < kGridDepth; ++z)
        {
            for (int x = 0; x < kGridWidth; ++x)
            {
                const int shapeIndex = (x + z + layer + static_cast<int>(envIndex)) % 3;
                const ColliderShapeType shape =
                    shapeIndex == 0 ? ColliderShapeType::Sphere
                                    : (shapeIndex == 1 ? ColliderShapeType::Box
                                                       : ColliderShapeType::Capsule);

                const auto entity = world.createEntity(envIndex);
                TransformComponent transform{};
                transform.worldTransform.position = {
                    envOrigin.x + xOrigin + static_cast<float>(x) * kSpacing,
                    kBaseHeight + static_cast<float>(layer) * kLayerHeight +
                        ((x + z + static_cast<int>(envIndex)) % 2 == 0 ? 0.0f : 0.22f),
                    envOrigin.z + zOrigin + static_cast<float>(z) * kSpacing};
                world.setTransform(entity, transform);

                MeshRendererComponent meshRenderer{};
                meshRenderer.mesh = meshForShape(shape);
                meshRenderer.material = materialForShape(shape);
                meshRenderer.visible = true;
                world.setMeshRenderer(entity, meshRenderer);

                RigidBodyComponent body{};
                body.simulated = true;
                body.inverseMass = 1.0f;
                body.inverseInertiaLocal =
                    inverseInertiaForShape(shape, colliderParamsForShape(shape), body.inverseMass);
                body.linearVelocity = {
                    static_cast<float>((x % 3) - 1) * 0.06f + envVelocityBiasX,
                    0.0f,
                    static_cast<float>((z % 3) - 1) * 0.06f + envVelocityBiasZ};
                body.angularVelocity = {0.0f, envAngularBias, 0.0f};
                world.setRigidBody(entity, body);

                cressim::neo::engine::ColliderComponent collider{};
                collider.shapeType = shape;
                collider.shapeParams = colliderParamsForShape(shape);
                collider.friction = 0.42f;
                collider.restitution = 0.03f;
                world.addCollider(entity, collider);
            }
        }
    }
}

EnvMaterialSet createMaterials(cressim::neo::graphics::RenderResourceManager &resources,
                               std::uint32_t envIndex)
{
    const EnvIblPalette palette = paletteForEnv(envIndex);

    MaterialResourceDesc mirrorSphere{};
    mirrorSphere.debugName = "ViewerIntegration.Ibl.MirrorSphere." + std::to_string(envIndex);
    mirrorSphere.baseColor = palette.sunColor;
    mirrorSphere.metallic = 1.0f;
    mirrorSphere.roughness = 0.03f;

    MaterialResourceDesc polishedBox{};
    polishedBox.debugName = "ViewerIntegration.Ibl.PolishedBox." + std::to_string(envIndex);
    polishedBox.baseColor = lerp(palette.accent, palette.horizon, 0.35f);
    polishedBox.metallic = 1.0f;
    polishedBox.roughness = 0.14f;

    MaterialResourceDesc roughMetalCapsule{};
    roughMetalCapsule.debugName =
        "ViewerIntegration.Ibl.RoughMetalCapsule." + std::to_string(envIndex);
    roughMetalCapsule.baseColor = lerp(palette.horizon, palette.ground, 0.30f);
    roughMetalCapsule.metallic = 1.0f;
    roughMetalCapsule.roughness = 0.34f;

    MaterialResourceDesc dielectricSphere{};
    dielectricSphere.debugName =
        "ViewerIntegration.Ibl.DielectricSphere." + std::to_string(envIndex);
    dielectricSphere.baseColor = lerp(palette.zenith, palette.accent, 0.18f);
    dielectricSphere.metallic = 0.0f;
    dielectricSphere.roughness = 0.08f;

    MaterialResourceDesc ground{};
    ground.debugName = "ViewerIntegration.Ibl.Ground." + std::to_string(envIndex);
    ground.baseColor = lerp(palette.averageRadiance, palette.ground, 0.55f);
    ground.metallic = 0.0f;
    ground.roughness = 0.92f;

    return {
        resources.registerMaterial(mirrorSphere),
        resources.registerMaterial(polishedBox),
        resources.registerMaterial(roughMetalCapsule),
        resources.registerMaterial(dielectricSphere),
        resources.registerMaterial(ground),
    };
}

bool assignEnvironmentIbl(cressim::neo::engine::World &world,
                          cressim::neo::graphics::RenderResourceManager &resources,
                          std::uint32_t envIndex)
{
    EnvironmentIblDesc ibl{};
    ibl.irradianceCubemap = resources.registerTexture(makeIrradianceCubeDesc(envIndex));
    ibl.prefilteredSpecularCubemap =
        resources.registerTexture(makePrefilteredSpecularCubeDesc(envIndex));
    ibl.intensity = 1.0f + 0.08f * static_cast<float>(envIndex % 3u);
    return world.setEnvironmentIbl(envIndex, ibl);
}

bool shouldAuthorEnvironmentIbl(const RuntimeConfig &config) noexcept
{
    return config.rendererDesc.iblQualityTier != cressim::neo::graphics::IblQualityTier::Off;
}

} // namespace

int main(int argc, char **argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.rendererDesc.iblQualityTier = cressim::neo::graphics::IblQualityTier::Full;
    std::uint64_t numFrames = 0u;
    std::uint32_t envCount = 4u;

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
        if (arg == "--envs")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            envCount = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            if (envCount == 0u)
            {
                CRESSIM_LOG_ERROR("--envs must be greater than zero.\n");
                return 2;
            }
            continue;
        }

        printUsage(argv[0]);
        return 2;
    }

    config.sceneLayout.envCount = envCount;
    config.sceneLayout.maxRenderableObjectsPerEnv = kObjectsPerEnvBudget;
    config.sceneLayout.maxLightsPerEnv = 1u;
    config.sceneLayout.maxCamerasPerEnv = 1u;

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled = windowEnabled;
    viewerDesc.windowVisible = windowEnabled;
    viewerDesc.startFullscreenWindowed = false;
    viewerDesc.maxFrames = numFrames;
    viewerDesc.showStats = true;
    viewerDesc.vSync = false;
    viewerDesc.width = 1280;
    viewerDesc.height = 720;
    viewerDesc.windowTitle = "CRESSim Neo Physics Viewer Large Array Multi Env IBL";

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

    auto &world = runtime.getWorld();
    auto &resources = runtime.getResources();

    const MeshHandle cubeMesh = resources.registerMesh(makeCubeMesh(0.45f));
    const MeshHandle planeMesh = resources.registerMesh(makePlaneMesh(28.0f));
    const MeshHandle sphereMesh = resources.registerMesh(makeSphereMesh(0.45f, 20u, 12u));
    const MeshHandle capsuleMesh =
        resources.registerMesh(makeCapsuleMesh(0.28f, 0.52f, 20u, 6u, 2u));

    cressim::neo::common::EntityId primaryCamera = cressim::neo::common::kInvalidEntityId;
    for (std::uint32_t envIndex = 0u; envIndex < envCount; ++envIndex)
    {
        if (shouldAuthorEnvironmentIbl(config) && !assignEnvironmentIbl(world, resources, envIndex))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("Failed to assign IBL for env ", envIndex, ".\n");
            return 1;
        }

        const EnvMaterialSet materials = createMaterials(resources, envIndex);
        cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(world, envIndex, envCount, cubeMesh, planeMesh, sphereMesh, capsuleMesh,
                          materials, cameraEntity);
        if (envIndex == 0u)
        {
            primaryCamera = cameraEntity;
        }
    }

    std::uint64_t beforeCalls = 0u;
    std::uint64_t afterCalls = 0u;

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&](const FrameContext &, Runtime &) { ++beforeCalls; };
    callbacks.afterTick = [&](const FrameContext &, Runtime &) { ++afterCalls; };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = primaryCamera;
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

    CRESSIM_LOG_INFO("Physics viewer large array multi-env IBL passed. Envs=", envCount,
                     " Frames=", viewerDesc.maxFrames, '\n');
    return 0;
}
