#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "viewer/debug_viewer_app.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

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
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::physics::SoftBodySourceKind;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCameraBinding;

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

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "SoftParticleToroidViewer.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3 &normal, const Diligent::float3 &v0,
                             const Diligent::float3 &v1, const Diligent::float3 &v2,
                             const Diligent::float3 &v3) {
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
    mesh.debugName = "SoftParticleToroidViewer.PlaneMesh";
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

std::filesystem::path fixturePath(const char *name)
{
    return std::filesystem::path(__FILE__).parent_path() / "fixtures" / name;
}

} // namespace

int main(int argc, char **argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.physicsDesc.softContactIterations  = 100;
    config.physicsDesc.softInternalIterations = 100;
    std::uint64_t numFrames = 0;

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

    const std::filesystem::path nodeFile = fixturePath("toroid.node");
    const std::filesystem::path eleFile  = fixturePath("toroid.ele");
    if (!std::filesystem::exists(nodeFile) || !std::filesystem::exists(eleFile))
    {
        CRESSIM_LOG_ERROR("Toroid TetGen fixtures are missing.");
        return 1;
    }

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled             = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled             = windowEnabled;
    viewerDesc.windowVisible             = windowEnabled;
    viewerDesc.startFullscreenWindowed   = windowEnabled;
    viewerDesc.maxFrames                 = numFrames;
    viewerDesc.showStats                 = true;
    viewerDesc.statsIntervalFrames       = 60u;
    viewerDesc.width                     = 1280;
    viewerDesc.height                    = 720;
    viewerDesc.enableDebugParticles      = true;

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

    auto &world               = runtime.getWorld();
    const auto cameraEntity   = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 18.0f, -42.0f};
    cameraTransform.worldTransform.rotation =
        Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f}, 0.34f);
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.35f, -1.0f, 0.25f};
    light.intensity = 7.0f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources           = runtime.getResources();
    const auto boxMesh        = resources.registerMesh(makeCubeMesh(2.5f));
    const auto planeMesh      = resources.registerMesh(makePlaneMesh(80.0f));

    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "SoftParticleToroidViewer.Ground";
    groundMaterialDesc.baseColor = {0.72f, 0.75f, 0.79f};
    groundMaterialDesc.metallic  = 0.0f;
    groundMaterialDesc.roughness = 0.90f;
    const auto groundMaterial    = resources.registerMaterial(groundMaterialDesc);

    MaterialResourceDesc obstacleMaterialDesc{};
    obstacleMaterialDesc.debugName = "SoftParticleToroidViewer.Obstacle";
    obstacleMaterialDesc.baseColor = {0.16f, 0.43f, 0.86f};
    obstacleMaterialDesc.metallic  = 0.0f;
    obstacleMaterialDesc.roughness = 0.55f;
    const auto obstacleMaterial    = resources.registerMaterial(obstacleMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -13.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, groundMaterial, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType   = ColliderShapeType::Box;
    groundCollider.shapeParams = {80.0f, 0.05f, 80.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    const auto obstacleEntity = world.createEntity();
    TransformComponent obstacleTransform{};
    obstacleTransform.worldTransform.position = {12.0f, 2.0f, 0.0f};
    world.setTransform(obstacleEntity, obstacleTransform);
    world.setMeshRenderer(obstacleEntity, MeshRendererComponent{boxMesh, obstacleMaterial, true});
    RigidBodyComponent obstacleBody{};
    obstacleBody.bodyType            = RigidBodyType::Dynamic;
    obstacleBody.inverseMass         = 0.02f;
    obstacleBody.inverseInertiaLocal = computeBoxInverseInertia({2.5f, 2.5f, 2.5f},
                                                                obstacleBody.inverseMass);
    world.setRigidBody(obstacleEntity, obstacleBody);
    ColliderComponent obstacleCollider{};
    obstacleCollider.shapeType   = ColliderShapeType::Box;
    obstacleCollider.shapeParams = {2.5f, 2.5f, 2.5f, 0.0f};
    world.addCollider(obstacleEntity, obstacleCollider);

    const auto staticObstacleEntity = world.createEntity();
    TransformComponent staticObstacleTransform{};
    staticObstacleTransform.worldTransform.position = {-10.0f, 1.5f, 0.0f};
    world.setTransform(staticObstacleEntity, staticObstacleTransform);
    world.setMeshRenderer(staticObstacleEntity,
                          MeshRendererComponent{boxMesh, obstacleMaterial, true});
    RigidBodyComponent staticObstacleBody{};
    staticObstacleBody.bodyType    = RigidBodyType::Static;
    staticObstacleBody.inverseMass = 0.0f;
    world.setRigidBody(staticObstacleEntity, staticObstacleBody);
    ColliderComponent staticObstacleCollider{};
    staticObstacleCollider.shapeType   = ColliderShapeType::Box;
    staticObstacleCollider.shapeParams = {2.5f, 2.5f, 2.5f, 0.0f};
    world.addCollider(staticObstacleEntity, staticObstacleCollider);

    const auto softEntity = world.createEntity();
    TransformComponent softTransform{};
    softTransform.worldTransform.position = {-2.0f, 10.0f, 0.0f};
    softTransform.worldTransform.scale    = {5.0f, 5.0f, 5.0f};
    world.setTransform(softEntity, softTransform);

    SoftBodyComponent softBody{};
    softBody.source.kind            = SoftBodySourceKind::TetGenFiles;
    softBody.source.tetGen.nodeFile = nodeFile.string();
    softBody.source.tetGen.eleFile  = eleFile.string();
    softBody.particleMass           = 0.02f;
    softBody.particleRadius         = 0.28f;
    softBody.edgeCompliance         = 0.0f;
    softBody.volumeCompliance       = 0.0001f;
    softBody.selfCollisionEnabled   = false;
    softBody.collisionLayer         = 0x1u;
    softBody.collisionMask          = 0xffffffffu;
    if (!world.setSoftBody(softEntity, softBody))
    {
        runtime.shutdown();
        viewer.shutdown();
        return 1;
    }

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{cameraEntity});

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
