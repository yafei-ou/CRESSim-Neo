#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/example_cli.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::PointLightComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::World;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::EnvironmentCubemapImage;
using cressim::neo::graphics::EnvironmentIblBakeOptions;
using cressim::neo::graphics::EnvironmentIblDesc;
using cressim::neo::graphics::IblQualityTier;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi = 3.14159265358979323846f;
constexpr int kGridWidth = 10;
constexpr int kGridDepth = 10;
constexpr int kLayers = 5;
constexpr float kSpacing = 1.4f;
constexpr float kBaseHeight = 1.5f;
constexpr float kLayerHeight = 2.0f;
constexpr float kEnvWorldSpacing = 72.0f;
constexpr std::uint32_t kDefaultEnvCount = 4u;
constexpr std::uint32_t kDynamicBodiesPerEnv =
    static_cast<std::uint32_t>(kGridWidth * kGridDepth * kLayers);
constexpr std::uint32_t kObjectsPerEnvBudget = kDynamicBodiesPerEnv + 16u;
constexpr std::uint32_t kMatrixObjectsPerEnvBudget = 24u;
constexpr std::uint32_t kSharedIrradianceSize = 16u;
constexpr std::uint32_t kSharedSpecularSize = 32u;
constexpr std::uint32_t kSharedSpecularMipCount = 6u;

enum class LightingMode
{
    Standard,
    Mixed,
    Matrix,
};

enum class IblMode
{
    None,
    PerEnv,
    Shared,
};

struct ExampleOptions
{
    CommonExampleOptions common{};
    LightingMode lightingMode = LightingMode::Standard;
    IblMode iblMode = IblMode::None;
};

struct EnvMaterialSet
{
    MaterialHandle box{};
    MaterialHandle sphere{};
    MaterialHandle capsule{};
    MaterialHandle ground{};
};

struct EnvPalette
{
    Diligent::float3 box{};
    Diligent::float3 sphere{};
    Diligent::float3 capsule{};
    Diligent::float3 ground{};
};

struct EnvironmentPalette
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

LightingMode parseLightingMode(const std::string& value)
{
    if (value == "standard")
    {
        return LightingMode::Standard;
    }
    if (value == "mixed")
    {
        return LightingMode::Mixed;
    }
    if (value == "matrix")
    {
        return LightingMode::Matrix;
    }

    throw std::invalid_argument("Unsupported lighting mode: " + value);
}

IblMode parseIblMode(const std::string& value)
{
    if (value == "none")
    {
        return IblMode::None;
    }
    if (value == "per_env")
    {
        return IblMode::PerEnv;
    }
    if (value == "shared")
    {
        return IblMode::Shared;
    }

    throw std::invalid_argument("Unsupported IBL mode: " + value);
}

void printUsage(const char* appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName, " [--lighting-mode standard|mixed|matrix] [--ibl-mode none|per_env|shared]",
        true);
}

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "LargeArray.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3& normal, const Diligent::float3& v0,
                             const Diligent::float3& v1, const Diligent::float3& v2,
                             const Diligent::float3& v3)
    {
        const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
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
    mesh.debugName = "LargeArray.PlaneMesh";
    const float h = halfExtent;
    mesh.vertices = {
        {{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f},
        {{h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 1.0f, 0.0f},
        {{h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 1.0f, 1.0f},
        {{-h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f},
    };
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return mesh;
}

MeshResourceDesc makeSphereMesh(float radius, std::uint32_t slices, std::uint32_t stacks)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "LargeArray.SphereMesh";
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

MeshResourceDesc makeCapsuleMesh(float radius, float halfHeight, std::uint32_t slices,
                                 std::uint32_t hemisphereRings, std::uint32_t bodyRings)
{
    struct Ring
    {
        float y = 0.0f;
        float r = 0.0f;
    };

    MeshResourceDesc mesh{};
    mesh.debugName = "LargeArray.CapsuleMesh";
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

Diligent::float3 computeBoxInverseInertia(const Diligent::float3& halfExtents, float inverseMass)
{
    if (inverseMass <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float ix =
        mass * (halfExtents.y * halfExtents.y + halfExtents.z * halfExtents.z) / 3.0f;
    const float iy =
        mass * (halfExtents.x * halfExtents.x + halfExtents.z * halfExtents.z) / 3.0f;
    const float iz =
        mass * (halfExtents.x * halfExtents.x + halfExtents.y * halfExtents.y) / 3.0f;

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
                                        const Diligent::float4& colliderParams,
                                        float inverseMass)
{
    switch (shape)
    {
    case ColliderShapeType::Sphere:
        return computeSphereInverseInertia(colliderParams.x, inverseMass);
    case ColliderShapeType::Box:
        return computeBoxInverseInertia(
            {colliderParams.x, colliderParams.y, colliderParams.z}, inverseMass);
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

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager& resources,
                                const std::string& name, const Diligent::float3& baseColor,
                                float roughness, float metallic = 0.0f)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.roughness = roughness;
    desc.metallic = metallic;
    return resources.registerMaterial(desc);
}

EnvPalette paletteForEnv(std::uint32_t envIndex)
{
    static constexpr std::array<EnvPalette, 6> kPalettes = {{
        {{0.85f, 0.28f, 0.18f}, {0.18f, 0.58f, 0.90f}, {0.28f, 0.82f, 0.36f}, {0.72f, 0.74f, 0.77f}},
        {{0.95f, 0.56f, 0.14f}, {0.14f, 0.76f, 0.90f}, {0.56f, 0.34f, 0.88f}, {0.80f, 0.75f, 0.63f}},
        {{0.82f, 0.18f, 0.48f}, {0.12f, 0.66f, 0.44f}, {0.94f, 0.82f, 0.20f}, {0.64f, 0.74f, 0.84f}},
        {{0.26f, 0.46f, 0.92f}, {0.92f, 0.24f, 0.24f}, {0.20f, 0.72f, 0.66f}, {0.66f, 0.68f, 0.78f}},
        {{0.84f, 0.34f, 0.16f}, {0.38f, 0.52f, 0.94f}, {0.16f, 0.78f, 0.30f}, {0.76f, 0.70f, 0.60f}},
        {{0.72f, 0.20f, 0.20f}, {0.20f, 0.76f, 0.88f}, {0.58f, 0.30f, 0.82f}, {0.70f, 0.78f, 0.70f}},
    }};
    return kPalettes[envIndex % kPalettes.size()];
}

EnvironmentPalette environmentPaletteForEnv(std::uint32_t envIndex)
{
    switch (envIndex % 4u)
    {
    case 0u:
        return {{0.20f, 0.38f, 0.92f},
                {0.95f, 0.58f, 0.22f},
                {0.12f, 0.09f, 0.07f},
                {0.25f, 0.72f, 1.15f},
                Diligent::normalize(Diligent::float3{0.62f, 0.74f, -0.28f}),
                {1.0f, 0.92f, 0.78f},
                {0.34f, 0.26f, 0.20f},
                8.5f};
    case 1u:
        return {{0.05f, 0.22f, 0.48f},
                {0.18f, 0.72f, 0.92f},
                {0.02f, 0.05f, 0.08f},
                {0.95f, 0.30f, 0.82f},
                Diligent::normalize(Diligent::float3{-0.38f, 0.86f, 0.34f}),
                {0.86f, 0.94f, 1.0f},
                {0.14f, 0.28f, 0.36f},
                7.0f};
    case 2u:
        return {{0.62f, 0.70f, 0.84f},
                {0.92f, 0.82f, 0.70f},
                {0.20f, 0.18f, 0.15f},
                {1.10f, 0.65f, 0.24f},
                Diligent::normalize(Diligent::float3{0.24f, 0.92f, 0.30f}),
                {1.0f, 0.88f, 0.70f},
                {0.46f, 0.40f, 0.34f},
                6.5f};
    default:
        return {{0.08f, 0.06f, 0.16f},
                {0.82f, 0.18f, 0.24f},
                {0.03f, 0.02f, 0.04f},
                {0.34f, 0.95f, 0.62f},
                Diligent::normalize(Diligent::float3{-0.55f, 0.78f, -0.30f}),
                {1.0f, 0.78f, 0.72f},
                {0.24f, 0.10f, 0.12f},
                9.0f};
    }
}

EnvMaterialSet createMaterials(cressim::neo::graphics::RenderResourceManager& resources,
                               std::uint32_t envIndex)
{
    const auto palette = paletteForEnv(envIndex);
    return {
        registerMaterial(resources, "LargeArray.Box." + std::to_string(envIndex), palette.box, 0.55f),
        registerMaterial(resources, "LargeArray.Sphere." + std::to_string(envIndex), palette.sphere,
                         0.35f, 0.08f),
        registerMaterial(resources, "LargeArray.Capsule." + std::to_string(envIndex), palette.capsule,
                         0.45f, 0.02f),
        registerMaterial(resources, "LargeArray.Ground." + std::to_string(envIndex), palette.ground,
                         0.90f),
    };
}

float saturate(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

Diligent::float3 lerp(const Diligent::float3& a, const Diligent::float3& b, float t)
{
    return {lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t)};
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

Diligent::float3 sampleEnvironmentRadiance(const EnvironmentPalette& palette,
                                           const Diligent::float3& dir, float roughness)
{
    const float up = saturate(dir.y * 0.5f + 0.5f);
    const float skyMix = std::pow(up, 0.55f);
    const float groundMix = std::pow(1.0f - up, 1.45f);
    const float horizonBand = std::pow(1.0f - std::fabs(dir.y), 2.25f);
    const float azimuth =
        0.5f + 0.5f * std::sin(std::atan2(dir.z, dir.x) * 3.0f + roughness * 1.7f);

    Diligent::float3 color = lerp(palette.horizon, palette.zenith, skyMix);
    color = lerp(color, palette.ground, groundMix * 0.85f);
    color += palette.accent *
             (horizonBand * (0.20f + 0.40f * azimuth) * (1.0f - 0.45f * roughness));

    const float sunAlignment = saturate(Diligent::dot(dir, palette.sunDirection));
    const float exponent = lerp(96.0f, 6.0f, roughness);
    const float sunGlow = std::pow(sunAlignment, exponent) * palette.sunIntensity;
    color += palette.sunColor * sunGlow;

    return lerp(color, palette.averageRadiance, roughness * roughness * 0.92f);
}

std::array<EnvironmentCubemapImage, 6u> makeEnvironmentFaces(std::uint32_t envIndex)
{
    std::array<EnvironmentCubemapImage, 6u> faces;
    const auto palette = environmentPaletteForEnv(envIndex);
    for (std::uint32_t face = 0u; face < 6u; ++face)
    {
        faces[face].width = kSharedSpecularSize;
        faces[face].height = kSharedSpecularSize;
        auto& pixels = faces[face].pixels;
        pixels.reserve(static_cast<std::size_t>(kSharedSpecularSize) * kSharedSpecularSize);
        for (std::uint32_t y = 0u; y < kSharedSpecularSize; ++y)
        {
            for (std::uint32_t x = 0u; x < kSharedSpecularSize; ++x)
            {
                const auto dir = cubeDirection(face, x, y, kSharedSpecularSize);
                const auto rgb = sampleEnvironmentRadiance(palette, dir, 0.0f);
                pixels.push_back({rgb.x, rgb.y, rgb.z, 1.0f});
            }
        }
    }
    return faces;
}

EnvironmentIblDesc createEnvironmentIbl(cressim::neo::graphics::RenderResourceManager& resources,
                                        std::uint32_t envIndex)
{
    EnvironmentIblBakeOptions options{};
    options.irradianceSize = kSharedIrradianceSize;
    options.specularSize = kSharedSpecularSize;
    options.specularMipCount = kSharedSpecularMipCount;
    options.intensity = 1.0f;
    return cressim::neo::graphics::createEnvironmentIblFromCubemapImages(
        resources, makeEnvironmentFaces(envIndex), options);
}

bool assignSharedEnvironmentIbl(World& world,
                                cressim::neo::graphics::RenderResourceManager& resources,
                                std::uint32_t envCount)
{
    const auto ibl = createEnvironmentIbl(resources, 0u);
    for (std::uint32_t envIndex = 0u; envIndex < envCount; ++envIndex)
    {
        if (!world.setEnvironmentIbl(envIndex, ibl))
        {
            return false;
        }
    }
    return true;
}

bool assignPerEnvironmentIbl(World& world,
                             cressim::neo::graphics::RenderResourceManager& resources,
                             std::uint32_t envCount)
{
    for (std::uint32_t envIndex = 0u; envIndex < envCount; ++envIndex)
    {
        if (!world.setEnvironmentIbl(envIndex, createEnvironmentIbl(resources, envIndex)))
        {
            return false;
        }
    }
    return true;
}

void authorCamera(World& world, std::uint32_t envIndex, const Diligent::float3& position,
                  EntityId& outCameraEntity)
{
    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = position;
    world.setTransform(outCameraEntity, cameraTransform);

    CameraComponent camera{};
    camera.verticalFovDegrees = 55.0f;
    camera.clearColor = true;
    camera.clearDepth = true;
    camera.renderOrder = static_cast<int>(envIndex);
    world.setCamera(outCameraEntity, camera);
}

void authorGround(World& world, std::uint32_t envIndex, const Diligent::float3& envOrigin,
                  MeshHandle planeMesh, MaterialHandle groundMaterial, float halfExtent)
{
    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = envOrigin + Diligent::float3{0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, groundMaterial, true});

    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = RigidBodyType::Static;
    world.setRigidBody(groundEntity, groundBody);

    cressim::neo::engine::ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {halfExtent, 0.05f, halfExtent, 0.0f};
    world.addCollider(groundEntity, groundCollider);
}

void authorDynamicArray(World& world, std::uint32_t envIndex, std::uint32_t envCount,
                        const Diligent::float3& envOrigin, MeshHandle cubeMesh,
                        MeshHandle sphereMesh, MeshHandle capsuleMesh,
                        const EnvMaterialSet& materials)
{
    const float envPhase = static_cast<float>(envIndex) * 0.45f;
    const float envVelocityBiasX = std::cos(envPhase) * 0.05f;
    const float envVelocityBiasZ = std::sin(envPhase) * 0.05f;
    const float envAngularBias = 0.20f + 0.05f * static_cast<float>(envIndex % 5u);
    const float envRestitution = 0.02f * static_cast<float>(envIndex % 4u);
    const float envFriction = 0.35f + 0.08f * static_cast<float>(envIndex % 4u);

    const auto meshForShape = [&](ColliderShapeType shape) {
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

    const auto materialForShape = [&](ColliderShapeType shape) {
        switch (shape)
        {
        case ColliderShapeType::Sphere:
            return materials.sphere;
        case ColliderShapeType::Box:
            return materials.box;
        case ColliderShapeType::Capsule:
            return materials.capsule;
        }
        return materials.box;
    };

    const float xOrigin = -0.5f * static_cast<float>(kGridWidth - 1) * kSpacing;
    const float zOrigin = -0.5f * static_cast<float>(kGridDepth - 1) * kSpacing;

    for (int layer = 0; layer < kLayers; ++layer)
    {
        for (int z = 0; z < kGridDepth; ++z)
        {
            for (int x = 0; x < kGridWidth; ++x)
            {
                const int shapeIndex = (x + z + layer + static_cast<int>(envIndex)) % 3;
                const auto shape = shapeIndex == 0 ? ColliderShapeType::Box
                                                   : (shapeIndex == 1 ? ColliderShapeType::Sphere
                                                                      : ColliderShapeType::Capsule);

                const auto entity = world.createEntity(envIndex);
                TransformComponent transform{};
                transform.worldTransform.position = {
                    envOrigin.x + xOrigin + static_cast<float>(x) * kSpacing,
                    kBaseHeight + static_cast<float>(layer) * kLayerHeight +
                        ((x + z + static_cast<int>(envIndex)) % 2 == 0 ? 0.0f : 0.25f) +
                        0.05f * static_cast<float>(envCount % 3u),
                    envOrigin.z + zOrigin + static_cast<float>(z) * kSpacing};
                world.setTransform(entity, transform);
                world.setMeshRenderer(
                    entity, MeshRendererComponent{meshForShape(shape), materialForShape(shape), true});

                RigidBodyComponent body{};
                body.simulated = true;
                body.inverseMass = 1.0f;
                body.inverseInertiaLocal =
                    inverseInertiaForShape(shape, colliderParamsForShape(shape), body.inverseMass);
                body.linearVelocity = {
                    static_cast<float>((x % 3) - 1) * 0.08f + envVelocityBiasX,
                    0.0f,
                    static_cast<float>((z % 3) - 1) * 0.08f + envVelocityBiasZ};
                body.angularVelocity = {0.0f, envAngularBias, 0.0f};
                world.setRigidBody(entity, body);

                cressim::neo::engine::ColliderComponent collider{};
                collider.shapeType = shape;
                collider.shapeParams = colliderParamsForShape(shape);
                collider.friction = envFriction;
                collider.restitution = envRestitution;
                world.addCollider(entity, collider);
            }
        }
    }
}

void authorStandardLighting(World& world, std::uint32_t envIndex, float envPhase)
{
    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = Diligent::normalize(
        Diligent::float3{-0.45f + 0.08f * std::sin(envPhase), -1.0f,
                         0.35f + 0.08f * std::cos(envPhase)});
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 8.0f;
    world.setDirectionalLight(lightEntity, light);
}

void authorMixedLighting(World& world, std::uint32_t envIndex, const Diligent::float3& envOrigin)
{
    const float envPhase = static_cast<float>(envIndex) * 0.55f;

    if (envIndex % 3u != 2u)
    {
        const auto lightEntity = world.createEntity(envIndex);
        DirectionalLightComponent light{};
        light.direction = Diligent::normalize(
            Diligent::float3{-0.55f + 0.12f * std::sin(envPhase), -1.0f,
                             0.30f + 0.10f * std::cos(envPhase)});
        light.color = (envIndex % 2u == 0u) ? Diligent::float3{1.0f, 0.95f, 0.88f}
                                            : Diligent::float3{0.72f, 0.86f, 1.0f};
        light.intensity = (envIndex % 2u == 0u) ? 7.5f : 5.5f;
        light.castsShadows = (envIndex % 3u == 0u);
        world.setDirectionalLight(lightEntity, light);
    }

    const auto pointEntity = world.createEntity(envIndex);
    TransformComponent pointTransform{};
    pointTransform.worldTransform.position =
        envOrigin + Diligent::float3{-6.0f + static_cast<float>(envIndex % 3u) * 2.4f, 5.0f, -2.5f};
    world.setTransform(pointEntity, pointTransform);
    PointLightComponent point{};
    point.color = (envIndex % 2u == 0u) ? Diligent::float3{1.0f, 0.42f, 0.30f}
                                        : Diligent::float3{0.34f, 0.64f, 1.0f};
    point.intensity = 28.0f;
    point.range = 14.0f;
    world.setPointLight(pointEntity, point);
}

void authorMatrixEnvironment(World& world, std::uint32_t envIndex, std::uint32_t envCount,
                             const Diligent::float3& envOrigin, MeshHandle cubeMesh,
                             MeshHandle sphereMesh, MeshHandle planeMesh,
                             const EnvMaterialSet& materials, EntityId& outCameraEntity)
{
    authorCamera(world, envIndex, envOrigin + Diligent::float3{0.0f, 5.4f, -18.0f}, outCameraEntity);
    authorGround(world, envIndex, envOrigin, planeMesh, materials.ground, 9.0f);

    const float spacing = 2.2f;
    const float xOrigin = -spacing;
    const float zOrigin = -spacing;
    for (std::uint32_t z = 0u; z < 3u; ++z)
    {
        for (std::uint32_t x = 0u; x < 3u; ++x)
        {
            const auto entity = world.createEntity(envIndex);
            TransformComponent transform{};
            transform.worldTransform.position = envOrigin + Diligent::float3{
                xOrigin + static_cast<float>(x) * spacing,
                ((x + z) % 2u == 0u) ? 0.3f : 0.75f,
                zOrigin + static_cast<float>(z) * spacing};
            transform.worldTransform.scale = ((x + z) % 2u == 0u)
                                                 ? Diligent::float3{0.9f, 0.9f, 0.9f}
                                                 : Diligent::float3{0.75f, 0.75f, 0.75f};
            world.setTransform(entity, transform);
            world.setMeshRenderer(entity,
                                  MeshRendererComponent{((x + z) % 2u == 0u) ? cubeMesh : sphereMesh,
                                                        ((x + z) % 2u == 0u) ? materials.box
                                                                             : materials.sphere,
                                                        true});
        }
    }

    authorMixedLighting(world, envIndex, envOrigin);

    const auto fillDirEntity = world.createEntity(envIndex);
    DirectionalLightComponent fillDir{};
    fillDir.direction = Diligent::normalize(Diligent::float3{0.65f, -0.4f, -0.2f});
    fillDir.color = {0.55f, 0.72f, 1.0f};
    fillDir.intensity = 1.8f;
    fillDir.castsShadows = true;
    world.setDirectionalLight(fillDirEntity, fillDir);
}

void authorLargeArrayEnvironment(World& world, std::uint32_t envIndex, std::uint32_t envCount,
                                 LightingMode lightingMode, MeshHandle cubeMesh,
                                 MeshHandle planeMesh, MeshHandle sphereMesh,
                                 MeshHandle capsuleMesh, const EnvMaterialSet& materials,
                                 EntityId& outCameraEntity)
{
    const auto envOrigin = envWorldOrigin(envIndex, envCount);
    if (lightingMode == LightingMode::Matrix)
    {
        authorMatrixEnvironment(world, envIndex, envCount, envOrigin, cubeMesh, sphereMesh,
                                planeMesh, materials, outCameraEntity);
        return;
    }

    authorCamera(world, envIndex, envOrigin + Diligent::float3{0.0f, 5.0f, -26.0f},
                 outCameraEntity);
    authorGround(world, envIndex, envOrigin, planeMesh, materials.ground, 28.0f);

    const float envPhase = static_cast<float>(envIndex) * 0.45f;
    if (lightingMode == LightingMode::Standard)
    {
        authorStandardLighting(world, envIndex, envPhase);
    }
    else
    {
        authorMixedLighting(world, envIndex, envOrigin);
    }

    authorDynamicArray(world, envIndex, envCount, envOrigin, cubeMesh, sphereMesh, capsuleMesh,
                       materials);
}

const char* lightingModeName(LightingMode mode)
{
    switch (mode)
    {
    case LightingMode::Standard:
        return "standard";
    case LightingMode::Mixed:
        return "mixed";
    case LightingMode::Matrix:
        return "matrix";
    }
    return "standard";
}

const char* iblModeName(IblMode mode)
{
    switch (mode)
    {
    case IblMode::None:
        return "none";
    case IblMode::PerEnv:
        return "per_env";
    case IblMode::Shared:
        return "shared";
    }
    return "none";
}

} // namespace

int main(int argc, char** argv)
{
    ExampleOptions options{};
    options.common.envCount = kDefaultEnvCount;

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options.common, true))
            {
                continue;
            }

            const std::string arg = argv[i];
            if (arg == "--lighting-mode")
            {
                options.lightingMode = parseLightingMode(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--lighting-mode"));
                continue;
            }
            if (arg == "--ibl-mode")
            {
                options.iblMode = parseIblMode(
                    cressim::neo::examples::helpers::requireOptionValue(argc, argv, i, "--ibl-mode"));
                continue;
            }

            printUsage(argv[0]);
            return 2;
        }
    }
    catch (const std::invalid_argument& error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    config.rendererDesc.iblQualityTier =
        (options.iblMode == IblMode::None) ? IblQualityTier::Off : IblQualityTier::Full;
    config.sceneLayout.envCount = options.common.envCount;
    config.sceneLayout.maxRenderableObjectsPerEnv =
        (options.lightingMode == LightingMode::Matrix) ? kMatrixObjectsPerEnvBudget
                                                       : kObjectsPerEnvBudget;
    config.sceneLayout.maxLightsPerEnv =
        (options.lightingMode == LightingMode::Matrix) ? 6u : 4u;
    config.sceneLayout.maxCamerasPerEnv = 1u;

    DebugViewerApp viewer;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(
        options.common, ViewerExampleDefaults{
                            .windowTitle = "CRESSim Neo Physics Viewer Large Array Multi Env",
                            .width = (options.lightingMode == LightingMode::Matrix) ? 960u : 1280u,
                            .height = (options.lightingMode == LightingMode::Matrix) ? 540u : 720u,
                            .showStats = true,
                            .vSync = false,
                            .startFullscreen = false,
                            .startFullscreenWindowed = false,
                        });

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

    auto& world = runtime.getWorld();
    auto& resources = runtime.getResources();

    if (options.iblMode == IblMode::Shared &&
        !assignSharedEnvironmentIbl(world, resources, options.common.envCount))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to assign shared IBL across environments.\n");
        return 1;
    }

    if (options.iblMode == IblMode::PerEnv &&
        !assignPerEnvironmentIbl(world, resources, options.common.envCount))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Failed to assign per-environment IBL.\n");
        return 1;
    }

    const auto cubeMesh = resources.registerMesh(makeCubeMesh(0.45f));
    const auto planeMesh = resources.registerMesh(
        makePlaneMesh((options.lightingMode == LightingMode::Matrix) ? 9.0f : 28.0f));
    const auto sphereMesh = resources.registerMesh(makeSphereMesh(0.45f, 20u, 12u));
    const auto capsuleMesh = resources.registerMesh(makeCapsuleMesh(0.28f, 0.52f, 20u, 6u, 2u));

    EntityId primaryCamera = cressim::neo::common::kInvalidEntityId;
    for (std::uint32_t envIndex = 0u; envIndex < options.common.envCount; ++envIndex)
    {
        EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        const auto materials = createMaterials(resources, envIndex);
        authorLargeArrayEnvironment(world, envIndex, options.common.envCount, options.lightingMode,
                                    cubeMesh, planeMesh, sphereMesh, capsuleMesh, materials,
                                    cameraEntity);
        if (envIndex == 0u)
        {
            primaryCamera = cameraEntity;
        }
    }

    std::uint64_t beforeCalls = 0u;
    std::uint64_t afterCalls = 0u;

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&](const FrameContext&, Runtime&) { ++beforeCalls; };
    callbacks.afterTick = [&](const FrameContext&, Runtime&) { ++afterCalls; };

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

    if (viewerDesc.maxFrames > 0u &&
        (beforeCalls != viewerDesc.maxFrames || afterCalls != viewerDesc.maxFrames))
    {
        CRESSIM_LOG_ERROR("Unexpected callback counts. before=", beforeCalls, " after=", afterCalls,
                          " expected=", viewerDesc.maxFrames, "\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Physics viewer large array multi-env passed. Envs=", options.common.envCount,
                     " lightingMode=", lightingModeName(options.lightingMode), " iblMode=",
                     iblModeName(options.iblMode), " Frames=", viewerDesc.maxFrames, "\n");
    return 0;
}
