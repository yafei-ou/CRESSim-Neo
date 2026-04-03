#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::SoftBodyComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kPi                      = 3.14159265358979323846f;
constexpr float kEnvSpacing              = 16.0f;
constexpr std::uint32_t kDefaultEnvCount = 4u;

struct SceneMaterials
{
    MaterialHandle ground{};
    MaterialHandle staticObstacle{};
    MaterialHandle dynamicObstacle{};
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
    CRESSIM_LOG_ERROR("Usage: ", appName, " [--backend vulkan|null] [--frames N] [--envs N]\n");
}

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "SoftParticleMultiEnv.CubeMesh";
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
    mesh.debugName = "SoftParticleMultiEnv.PlaneMesh";
    const float h  = halfExtent;
    mesh.vertices  = {{{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f},
                      {{h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 1.0f, 0.0f},
                      {{h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 1.0f, 1.0f},
                      {{-h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f}};
    mesh.indices   = {0u, 1u, 2u, 0u, 2u, 3u};
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

Diligent::float3 envOrigin(std::uint32_t envIndex, std::uint32_t envCount)
{
    const std::uint32_t cols = std::max(
        1u, static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<float>(envCount)))));
    const std::uint32_t rows = std::max(1u, (envCount + cols - 1u) / cols);
    const std::uint32_t col  = envIndex % cols;
    const std::uint32_t row  = envIndex / cols;
    const float xCenter      = (static_cast<float>(cols) - 1.0f) * 0.5f;
    const float zCenter      = (static_cast<float>(rows) - 1.0f) * 0.5f;
    return {(static_cast<float>(col) - xCenter) * kEnvSpacing, 0.0f,
            (static_cast<float>(row) - zCenter) * kEnvSpacing};
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic  = 0.0f;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

struct SoftCollisionScenario
{
    std::uint32_t primaryLayer         = 0x1u;
    std::uint32_t primaryMask          = 0x3u;
    bool primarySelfCollisionEnabled   = true;
    std::uint32_t secondaryLayer       = 0x2u;
    std::uint32_t secondaryMask        = 0x1u;
    bool secondarySelfCollisionEnabled = false;
};

SoftCollisionScenario makeSoftCollisionScenario(std::uint32_t envIndex)
{
    switch (envIndex % 4u)
    {
    case 0u:
        // Cross-body collision enabled, primary self-collides, secondary does not.
        return {0x1u, 0x3u, true, 0x2u, 0x1u, false};
    case 1u:
        // Masks reject cross-body collision entirely.
        return {0x1u, 0x1u, true, 0x2u, 0x2u, false};
    case 2u:
        // Cross-body collision enabled and both bodies self-collide.
        return {0x1u, 0x3u, true, 0x2u, 0x1u, true};
    default:
        // Cross-body collision enabled and neither body self-collides.
        return {0x1u, 0x3u, false, 0x2u, 0x1u, false};
    }
}

void authorEnvironment(Runtime &runtime, std::uint32_t envIndex, std::uint32_t envCount,
                       MeshHandle boxMesh, MeshHandle planeMesh, const SceneMaterials &materials,
                       cressim::neo::common::EntityId &outCameraEntity)
{
    auto &world                   = runtime.getWorld();
    const Diligent::float3 origin = envOrigin(envIndex, envCount);
    const float phase             = static_cast<float>(envIndex) * 0.73f;

    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position =
        origin + Diligent::float3{0.0f, 3.8f, -8.8f - 0.35f * static_cast<float>(envIndex % 3u)};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.24f);
    world.setTransform(outCameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 50.0f;
    camera.renderOrder        = envIndex;
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = Diligent::normalize(
        Diligent::float3{-0.35f + 0.10f * std::sin(phase), -1.0f, 0.25f + 0.08f * std::cos(phase)});
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = origin + Diligent::float3{0.0f, -1.1f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, materials.ground, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType   = ColliderShapeType::Box;
    groundCollider.shapeParams = {6.0f, 0.05f, 6.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    const auto dynamicObstacleEntity = world.createEntity(envIndex);
    TransformComponent dynamicTransform{};
    dynamicTransform.worldTransform.position =
        origin + Diligent::float3{1.4f + 0.35f * std::sin(phase), 0.7f, -0.3f};
    dynamicTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.2f * phase);
    world.setTransform(dynamicObstacleEntity, dynamicTransform);
    world.setMeshRenderer(dynamicObstacleEntity,
                          MeshRendererComponent{boxMesh, materials.dynamicObstacle, true});
    RigidBodyComponent dynamicBody{};
    dynamicBody.bodyType    = RigidBodyType::Dynamic;
    dynamicBody.inverseMass = 0.65f + 0.10f * static_cast<float>(envIndex % 3u);
    dynamicBody.inverseInertiaLocal =
        computeBoxInverseInertia({0.55f, 0.45f, 0.55f}, dynamicBody.inverseMass);
    dynamicBody.linearVelocity = {-0.15f + 0.05f * std::cos(phase), 0.0f, 0.04f * std::sin(phase)};
    world.setRigidBody(dynamicObstacleEntity, dynamicBody);
    ColliderComponent dynamicCollider{};
    dynamicCollider.shapeType   = ColliderShapeType::Box;
    dynamicCollider.shapeParams = {0.55f, 0.45f, 0.55f, 0.0f};
    world.addCollider(dynamicObstacleEntity, dynamicCollider);

    const auto staticObstacleEntity = world.createEntity(envIndex);
    TransformComponent staticTransform{};
    staticTransform.worldTransform.position = origin + Diligent::float3{-1.15f, 0.55f, 0.2f};
    staticTransform.worldTransform.rotation = Diligent::QuaternionF::RotationFromAxisAngle(
        {0.0f, 0.0f, 1.0f}, 0.12f * static_cast<float>((envIndex % 5u) - 2u));
    world.setTransform(staticObstacleEntity, staticTransform);
    world.setMeshRenderer(staticObstacleEntity,
                          MeshRendererComponent{boxMesh, materials.staticObstacle, true});
    RigidBodyComponent staticBody{};
    staticBody.bodyType    = RigidBodyType::Static;
    staticBody.inverseMass = 0.0f;
    world.setRigidBody(staticObstacleEntity, staticBody);
    ColliderComponent staticCollider{};
    staticCollider.shapeType   = ColliderShapeType::Box;
    staticCollider.shapeParams = {0.7f, 0.28f, 0.7f, 0.0f};
    world.addCollider(staticObstacleEntity, staticCollider);

    const auto addSoftBody = [&](const cressim::neo::common::EntityId softEntityId,
                                 const SoftBodyComponent &softBody,
                                 const TransformComponent &softTransform)
    {
        world.setTransform(softEntityId, softTransform);
        world.setSoftBody(softEntityId, softBody);
    };

    const SoftCollisionScenario collisionScenario = makeSoftCollisionScenario(envIndex);

    TransformComponent primarySoftTransform{};
    primarySoftTransform.worldTransform.position =
        origin + Diligent::float3{-0.4f + 0.35f * std::cos(phase), 1.55f + 0.15f * std::sin(phase),
                                  -0.5f + 0.20f * std::cos(phase * 0.5f)};
    const Diligent::QuaternionF tiltX = Diligent::QuaternionF::RotationFromAxisAngle(
        {1.0f, 0.0f, 0.0f}, -0.18f + 0.06f * std::sin(phase));
    const Diligent::QuaternionF tiltY =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.25f * std::cos(phase));
    primarySoftTransform.worldTransform.rotation = tiltY * tiltX;
    SoftBodyComponent primarySoftBody{};
    primarySoftBody.size                 = {1.2f, 1.2f, 1.2f};
    primarySoftBody.particleSpacing      = 0.32f;
    primarySoftBody.particleMass         = 0.14f;
    primarySoftBody.particleRadius       = 0.16f;
    primarySoftBody.volumeCompliance     = 0.0010f;
    primarySoftBody.edgeCompliance       = 0.0f;
    primarySoftBody.selfCollisionEnabled = collisionScenario.primarySelfCollisionEnabled;
    primarySoftBody.collisionLayer       = collisionScenario.primaryLayer;
    primarySoftBody.collisionMask        = collisionScenario.primaryMask;
    addSoftBody(world.createEntity(envIndex), primarySoftBody, primarySoftTransform);

    TransformComponent secondarySoftTransform{};
    constexpr float kSoftBodyGap = 1.0f;
    secondarySoftTransform.worldTransform.position =
        primarySoftTransform.worldTransform.position +
        Diligent::float3{0.0f, primarySoftBody.size.y + kSoftBodyGap, 0.0f};
    secondarySoftTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, -0.2f + 0.05f * phase);
    SoftBodyComponent secondarySoftBody{};
    secondarySoftBody.size            = {0.9f, 0.9f, 0.9f};
    secondarySoftBody.particleSpacing = 0.30f;
    secondarySoftBody.particleMass    = 0.10f;
    secondarySoftBody.particleRadius =
        0.20f; // 0.15 easily leave gaps; manually fatten the collision range
    secondarySoftBody.volumeCompliance     = 0.0015f;
    secondarySoftBody.edgeCompliance       = 0.0f;
    secondarySoftBody.selfCollisionEnabled = collisionScenario.secondarySelfCollisionEnabled;
    secondarySoftBody.collisionLayer       = collisionScenario.secondaryLayer;
    secondarySoftBody.collisionMask        = collisionScenario.secondaryMask;
    addSoftBody(world.createEntity(envIndex), secondarySoftBody, secondarySoftTransform);
}

} // namespace

int main(int argc, char **argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend     = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation     = false;
    config.physicsDesc.softContactIterations  = 100;
    config.physicsDesc.softInternalIterations = 100;
    std::uint64_t numFrames                   = 0u;
    std::uint32_t envCount                    = kDefaultEnvCount;

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
                printUsage(argv[0]);
                return 2;
            }
            continue;
        }

        printUsage(argv[0]);
        return 2;
    }

    config.sceneLayout.envCount = envCount;

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled = windowEnabled;
    viewerDesc.windowVisible = windowEnabled;
    viewerDesc.startFullscreenWindowed = windowEnabled;
    viewerDesc.maxFrames               = numFrames;
    viewerDesc.showStats               = true;
    viewerDesc.statsIntervalFrames     = 60u;
    viewerDesc.width                   = 1600;
    viewerDesc.height                  = 900;
    viewerDesc.enableDebugParticles    = true;

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

    auto &resources            = runtime.getResources();
    const MeshHandle boxMesh   = resources.registerMesh(makeCubeMesh(0.6f));
    const MeshHandle planeMesh = resources.registerMesh(makePlaneMesh(6.0f));

    SceneMaterials materials{};
    materials.ground =
        registerMaterial(resources, "SoftParticleMultiEnv.Ground", {0.72f, 0.75f, 0.79f}, 0.90f);
    materials.staticObstacle  = registerMaterial(resources, "SoftParticleMultiEnv.StaticObstacle",
                                                 {0.15f, 0.43f, 0.85f}, 0.58f);
    materials.dynamicObstacle = registerMaterial(resources, "SoftParticleMultiEnv.DynamicObstacle",
                                                 {0.78f, 0.25f, 0.20f}, 0.42f);

    cressim::neo::common::EntityId primaryCamera = cressim::neo::common::kInvalidEntityId;

    for (std::uint32_t envIndex = 0u; envIndex < envCount; ++envIndex)
    {
        cressim::neo::common::EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(runtime, envIndex, envCount, boxMesh, planeMesh, materials, cameraEntity);
        if (envIndex == 0u)
        {
            primaryCamera = cameraEntity;
        }
    }

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{primaryCamera});

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
