#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "graphics/environment_ibl_baker.h"
#include "helpers/example_cli.h"
#include "helpers/inertia.h"
#include "helpers/shape_meshes.h"
#include "helpers/skybox_example.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

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

constexpr int kGridWidth = 10;
constexpr int kGridDepth = 10;
constexpr int kLayers = 5;
constexpr float kSpacing = 1.4f;
constexpr float kBaseHeight = 1.5f;
constexpr float kLayerHeight = 2.0f;
constexpr float kEnvWorldSpacing = 72.0f;
constexpr float kLargeArrayGroundHalfExtent = 28.0f;
constexpr float kContainerWallThickness = 0.25f;
constexpr float kContainerWallHalfHeight = 2.5f;
constexpr float kContainerInnerPadding = 0.7f;
constexpr std::uint32_t kDefaultEnvCount = 4u;
constexpr std::uint32_t kDynamicBodiesPerEnv =
    static_cast<std::uint32_t>(kGridWidth * kGridDepth * kLayers);
constexpr std::uint32_t kObjectsPerEnvBudget = kDynamicBodiesPerEnv + 16u;
constexpr std::uint32_t kMatrixObjectsPerEnvBudget = 24u;
constexpr std::uint32_t kSharedIrradianceSize = 16u;
constexpr std::array<const char*, 25u> kLargeArraySkyboxCrossPaths = {{
    "examples/cubemaps/Cubemap/Cubemap_Sky_01-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_02-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_03-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_04-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_05-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_06-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_07-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_08-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_09-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_10-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_11-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_12-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_13-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_14-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_15-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_16-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_17-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_18-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_19-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_20-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_21-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_22-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_23-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_24-512x512.png",
    "examples/cubemaps/Cubemap/Cubemap_Sky_25-512x512.png",
}};

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

enum class SkyboxBackgroundMode
{
    Off,
    On,
};

enum class ContainerMode
{
    Off,
    On,
};

struct ExampleOptions
{
    CommonExampleOptions common{};
    LightingMode lightingMode = LightingMode::Standard;
    IblMode iblMode = IblMode::None;
    SkyboxBackgroundMode skyboxBackgroundMode = SkyboxBackgroundMode::Off;
    ContainerMode containerMode = ContainerMode::Off;
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
        appName,
        " [--lighting-mode standard|mixed|matrix] [--ibl-mode none|per_env|shared]"
        " [--skybox-background off|on] [--container off|on]",
        true);
}

SkyboxBackgroundMode parseSkyboxBackgroundMode(const std::string& value)
{
    if (value == "off")
    {
        return SkyboxBackgroundMode::Off;
    }
    if (value == "on")
    {
        return SkyboxBackgroundMode::On;
    }

    throw std::invalid_argument("Unsupported skybox background mode: " + value);
}

ContainerMode parseContainerMode(const std::string& value)
{
    if (value == "off")
    {
        return ContainerMode::Off;
    }
    if (value == "on")
    {
        return ContainerMode::On;
    }

    throw std::invalid_argument("Unsupported container mode: " + value);
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

EnvironmentIblDesc loadFileSkyboxIbl(cressim::neo::graphics::RenderResourceManager& resources,
                                     std::uint32_t skyboxIndex,
                                     float intensity, float backgroundIntensity)
{
    const std::filesystem::path crossPath =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
        kLargeArraySkyboxCrossPaths[skyboxIndex % kLargeArraySkyboxCrossPaths.size()];

    EnvironmentIblBakeOptions options{};
    options.irradianceSize = kSharedIrradianceSize;
    options.specularSize = 128u;
    options.specularMipCount = 7u;
    options.irradianceSampleCount = 256u;
    options.specularSampleCount = 128u;
    options.intensity = intensity;
    options.backgroundIntensity = backgroundIntensity;
    return cressim::neo::examples::helpers::createEnvironmentIblFromHorizontalCross(
        resources, crossPath, options);
}

bool assignSharedEnvironmentIbl(World& world,
                                cressim::neo::graphics::RenderResourceManager& resources,
                                std::uint32_t envCount)
{
    const auto ibl = loadFileSkyboxIbl(resources, 0u, 0.25f, 1.0f);
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
        const auto ibl = loadFileSkyboxIbl(resources, envIndex, 0.25f, 1.0f);
        if (!world.setEnvironmentIbl(envIndex, ibl))
        {
            return false;
        }
    }
    return true;
}

void authorCamera(World& world, std::uint32_t envIndex, const Diligent::float3& position,
                  float pitchRadians, SkyboxBackgroundMode skyboxBackgroundMode,
                  EntityId& outCameraEntity)
{
    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = position;
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, pitchRadians);
    world.setTransform(outCameraEntity, cameraTransform);

    CameraComponent camera{};
    camera.verticalFovDegrees = 55.0f;
    camera.clearColor = true;
    camera.clearDepth = true;
    camera.renderOrder = static_cast<int>(envIndex);
    if (skyboxBackgroundMode == SkyboxBackgroundMode::On)
    {
        camera.backgroundMode = CameraComponent::BackgroundMode::EnvironmentCubemap;
    }
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
    groundBody.bodyType = RigidBodyType::Static;
    world.setRigidBody(groundEntity, groundBody);

    cressim::neo::engine::ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {halfExtent, 0.05f, halfExtent, 0.0f};
    groundCollider.friction = 0.1f;
    groundCollider.staticFriction = 0.2f;
    groundCollider.restitution = 0.5f;
    world.addCollider(groundEntity, groundCollider);
}

void authorContainerWalls(World& world, std::uint32_t envIndex, const Diligent::float3& envOrigin,
                          MeshHandle wallMesh, MaterialHandle wallMaterial, float halfExtent,
                          ContainerMode containerMode)
{
    constexpr float kMaxDynamicShapeHalfExtent = 0.52f;
    const float arrayHalfWidth = 0.5f * static_cast<float>(kGridWidth - 1) * kSpacing;
    const float arrayHalfDepth = 0.5f * static_cast<float>(kGridDepth - 1) * kSpacing;
    const float innerHalfWidth =
        std::min(arrayHalfWidth + kMaxDynamicShapeHalfExtent + kContainerInnerPadding,
                 halfExtent - kContainerWallThickness);
    const float innerHalfDepth =
        std::min(arrayHalfDepth + kMaxDynamicShapeHalfExtent + kContainerInnerPadding,
                 halfExtent - kContainerWallThickness);
    const float groundTopY = -1.0f + 0.05f;
    const float wallCenterY = groundTopY + kContainerWallHalfHeight;
    const Diligent::float3 wallScaleEW{innerHalfWidth + kContainerWallThickness,
                                       kContainerWallHalfHeight, kContainerWallThickness};
    const Diligent::float3 wallScaleNS{kContainerWallThickness, kContainerWallHalfHeight,
                                       innerHalfDepth + kContainerWallThickness};

    const auto authorWall = [&](const Diligent::float3& position,
                                const Diligent::float3& halfScale) {
        const auto wallEntity = world.createEntity(envIndex);

        TransformComponent wallTransform{};
        wallTransform.worldTransform.position = position;
        wallTransform.worldTransform.scale = halfScale;
        world.setTransform(wallEntity, wallTransform);
        world.setMeshRenderer(wallEntity, MeshRendererComponent{wallMesh, wallMaterial, true});

        RigidBodyComponent wallBody{};
        wallBody.bodyType = RigidBodyType::Static;
        world.setRigidBody(wallEntity, wallBody);

        cressim::neo::engine::ColliderComponent wallCollider{};
        wallCollider.shapeType = ColliderShapeType::Box;
        wallCollider.shapeParams = {1.0f, 1.0f, 1.0f, 0.0f};
        wallCollider.friction = 0.0f;
        wallCollider.staticFriction = 0.0f;
        wallCollider.restitution = 0.0f;
        world.addCollider(wallEntity, wallCollider);
    };

    authorWall(envOrigin + Diligent::float3{0.0f, wallCenterY,
                                            innerHalfDepth + kContainerWallThickness},
               wallScaleEW);
    authorWall(envOrigin + Diligent::float3{0.0f, wallCenterY,
                                            -(innerHalfDepth + kContainerWallThickness)},
               wallScaleEW);
    authorWall(envOrigin + Diligent::float3{innerHalfWidth + kContainerWallThickness, wallCenterY,
                                            0.0f},
               wallScaleNS);
    authorWall(envOrigin + Diligent::float3{-(innerHalfWidth + kContainerWallThickness), wallCenterY,
                                            0.0f},
               wallScaleNS);
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
    const float envRestitution = 0.2f * static_cast<float>(envIndex % 4u);
    const float envFriction = 0.05f + 0.15f * static_cast<float>(envIndex % 4u);

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
                body.inverseMass = 1.0f;
                body.inverseInertiaLocal =
                    cressim::neo::examples::helpers::computeInverseInertiaForShape(
                        shape, colliderParamsForShape(shape), body.inverseMass);
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
                collider.staticFriction = envFriction + 0.1f;
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
                             const EnvMaterialSet& materials,
                             SkyboxBackgroundMode skyboxBackgroundMode,
                             EntityId& outCameraEntity)
{
    authorCamera(world, envIndex, envOrigin + Diligent::float3{0.0f, 5.4f, -18.0f}, 0.18f,
                 skyboxBackgroundMode, outCameraEntity);
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
                                 LightingMode lightingMode,
                                 SkyboxBackgroundMode skyboxBackgroundMode,
                                 ContainerMode containerMode,
                                 MeshHandle cubeMesh, MeshHandle wallMesh, MeshHandle planeMesh, MeshHandle sphereMesh,
                                 MeshHandle capsuleMesh, const EnvMaterialSet& materials,
                                 EntityId& outCameraEntity)
{
    const auto envOrigin = envWorldOrigin(envIndex, envCount);
    if (lightingMode == LightingMode::Matrix)
    {
        authorMatrixEnvironment(world, envIndex, envCount, envOrigin, cubeMesh, sphereMesh,
                                planeMesh, materials, skyboxBackgroundMode, outCameraEntity);
        return;
    }

    authorCamera(world, envIndex, envOrigin + Diligent::float3{0.0f, 7.0f, -24.0f}, 0.30f,
                 skyboxBackgroundMode, outCameraEntity);
    authorGround(world, envIndex, envOrigin, planeMesh, materials.ground, kLargeArrayGroundHalfExtent);
    if (containerMode == ContainerMode::On)
    {
        authorContainerWalls(world, envIndex, envOrigin, wallMesh, materials.ground,
                             kLargeArrayGroundHalfExtent, containerMode);
    }

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
            if (arg == "--skybox-background")
            {
                options.skyboxBackgroundMode = parseSkyboxBackgroundMode(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--skybox-background"));
                continue;
            }
            if (arg == "--container")
            {
                options.containerMode = parseContainerMode(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--container"));
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
    config.physicsDesc.rigidRigidContactIterations = 20;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Physics Viewer Large Array Multi Env";
    viewerDefaults.showStats = true;
    auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(options.common, viewerDefaults);

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

    if (options.skyboxBackgroundMode == SkyboxBackgroundMode::On &&
        options.iblMode == IblMode::None)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("--skybox-background on requires --ibl-mode per_env or shared.\n");
        return 2;
    }

    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(0.45f, "LargeArray.CubeMesh"));
    const auto wallMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(1.0f, "LargeArray.WallMesh"));
    const auto planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(
            (options.lightingMode == LightingMode::Matrix) ? 9.0f : 28.0f,
            "LargeArray.PlaneMesh"));
    const auto sphereMesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        0.45f, 20u, 12u, "LargeArray.SphereMesh"));
    const auto capsuleMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCapsuleMesh(
            0.28f, 0.52f, 20u, 6u, 2u, "LargeArray.CapsuleMesh"));

    EntityId primaryCamera = cressim::neo::common::kInvalidEntityId;
    for (std::uint32_t envIndex = 0u; envIndex < options.common.envCount; ++envIndex)
    {
        EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        const auto materials = createMaterials(resources, envIndex);
        authorLargeArrayEnvironment(world, envIndex, options.common.envCount,
                                    options.lightingMode, options.skyboxBackgroundMode,
                                    options.containerMode, cubeMesh, wallMesh, planeMesh,
                                    sphereMesh, capsuleMesh, materials, cameraEntity);
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
