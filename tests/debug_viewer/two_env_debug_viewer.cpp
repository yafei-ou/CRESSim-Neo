#include "engine/components.h"
#include "engine/runtime.h"
#include "viewer/debug_viewer_app.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
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
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCameraBinding;

GpuBackend parseBackend(const std::string& value)
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

void printUsage(const char* appName)
{
    std::cerr << "Usage: " << appName << " [--backend vulkan|null] [--frames N]\n";
}

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "TwoEnvViewer.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace =
        [&](const Diligent::float3& normal, const Diligent::float3& v0,
            const Diligent::float3& v1, const Diligent::float3& v2, const Diligent::float3& v3)
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
    mesh.debugName = "TwoEnvViewer.PlaneMesh";
    const float h  = halfExtent;
    mesh.vertices  = {
        {{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f},
        {{h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 1.0f, 0.0f},
        {{h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 1.0f, 1.0f},
        {{-h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f}};
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
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

void authorEnvironment(cressim::neo::engine::World& world,
                       cressim::neo::graphics::RenderResourceManager& resources,
                       std::uint32_t envIndex, float viewportX, const Diligent::float3& cameraPos,
                       cressim::neo::graphics::MeshHandle cubeMesh,
                       cressim::neo::graphics::MeshHandle planeMesh,
                       cressim::neo::graphics::MaterialHandle groundMaterial,
                       cressim::neo::graphics::MaterialHandle cubeMaterialA,
                       cressim::neo::graphics::MaterialHandle cubeMaterialB,
                       cressim::neo::common::EntityId& outCameraEntity,
                       bool firstCubeIsRigidBody, int renderOrder = 0)
{
    (void)resources;
    outCameraEntity = world.createEntity(envIndex);
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = cameraPos;
    world.setTransform(outCameraEntity, cameraTransform);

    CameraComponent camera{};
    camera.viewport = {viewportX, 0.0f, 0.5f, 1.0f};
    camera.clearColor = (viewportX == 0.0f);
    camera.clearDepth = (viewportX == 0.0f);
    camera.renderOrder = renderOrder;
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color     = {1.0f, 1.0f, 1.0f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent ground{};
    ground.mesh     = planeMesh;
    ground.material = groundMaterial;
    ground.visible  = true;
    world.setMeshRenderer(groundEntity, ground);
    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = cressim::neo::physics::RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
    groundCollider.shapeParams = {8.0f, 0.05f, 8.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    const auto cubeAEntity = world.createEntity(envIndex);
    TransformComponent cubeATransform{};
    cubeATransform.worldTransform.position = {-0.8f, 0.3f, 0.2f};
    world.setTransform(cubeAEntity, cubeATransform);
    MeshRendererComponent cubeARenderer{};
    cubeARenderer.mesh     = cubeMesh;
    cubeARenderer.material = cubeMaterialA;
    cubeARenderer.visible  = true;
    world.setMeshRenderer(cubeAEntity, cubeARenderer);
    if (firstCubeIsRigidBody)
    {
        RigidBodyComponent cubeABody{};
        cubeABody.simulated = true;
        cubeABody.inverseMass = 1.0f;
        cubeABody.inverseInertiaLocal =
            computeBoxInverseInertia({0.65f, 0.65f, 0.65f}, cubeABody.inverseMass);
        world.setRigidBody(cubeAEntity, cubeABody);
        ColliderComponent cubeACollider{};
        cubeACollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
        cubeACollider.shapeParams = {0.65f, 0.65f, 0.65f, 0.0f};
        world.addCollider(cubeAEntity, cubeACollider);
    }

    const auto cubeBEntity = world.createEntity(envIndex);
    TransformComponent cubeBTransform{};
    cubeBTransform.worldTransform.position = {1.0f, 0.85f, 1.3f};
    cubeBTransform.worldTransform.scale    = {1.15f, 1.15f, 1.15f};
    world.setTransform(cubeBEntity, cubeBTransform);
    MeshRendererComponent cubeBRenderer{};
    cubeBRenderer.mesh     = cubeMesh;
    cubeBRenderer.material = cubeMaterialB;
    cubeBRenderer.visible  = true;
    world.setMeshRenderer(cubeBEntity, cubeBRenderer);
    if (!firstCubeIsRigidBody)
    {
        RigidBodyComponent cubeBBody{};
        cubeBBody.simulated = true;
        cubeBBody.inverseMass = 1.0f;
        cubeBBody.inverseInertiaLocal =
            computeBoxInverseInertia({0.65f * 1.15f, 0.65f * 1.15f, 0.65f * 1.15f},
                                     cubeBBody.inverseMass);
        world.setRigidBody(cubeBEntity, cubeBBody);
        ColliderComponent cubeBCollider{};
        cubeBCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
        cubeBCollider.shapeParams = {0.65f * 1.15f, 0.65f * 1.15f, 0.65f * 1.15f, 0.0f};
        world.addCollider(cubeBEntity, cubeBCollider);
    }
}

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.sceneLayout.envCount           = 2u;
    config.sceneLayout.maxObjectsPerEnv   = 8u;
    config.sceneLayout.maxLightsPerEnv    = 2u;
    config.sceneLayout.maxCamerasPerEnv   = 2u;
    std::uint64_t numFrames               = 0u;

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
    viewerDesc.startFullscreenWindowed = true;
    viewerDesc.maxFrames = numFrames;
    viewerDesc.showStats = false;
    viewerDesc.width = 1280;
    viewerDesc.height = 720;
    viewerDesc.windowTitle = "CRESSim Neo Two Env Viewer";

    if (!viewer.initialize(viewerDesc, config))
    {
        std::cerr << "Viewer initialization failed.\n";
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        std::cerr << "Runtime initialization failed.\n";
        return 1;
    }

    auto& world = runtime.getWorld();
    auto& resources = runtime.getResources();

    const auto cubeMesh = resources.registerMesh(makeCubeMesh(0.65f));
    const auto planeMesh = resources.registerMesh(makePlaneMesh(8.0f));

    MaterialResourceDesc env0GroundDesc{};
    env0GroundDesc.debugName = "TwoEnv.Env0Ground";
    env0GroundDesc.baseColor = {0.75f, 0.73f, 0.72f};
    env0GroundDesc.roughness = 0.9f;
    const auto env0GroundMaterial = resources.registerMaterial(env0GroundDesc);

    MaterialResourceDesc env1GroundDesc{};
    env1GroundDesc.debugName = "TwoEnv.Env1Ground";
    env1GroundDesc.baseColor = {0.70f, 0.78f, 0.92f};
    env1GroundDesc.roughness = 0.9f;
    const auto env1GroundMaterial = resources.registerMaterial(env1GroundDesc);

    MaterialResourceDesc env0RedDesc{};
    env0RedDesc.debugName = "TwoEnv.Env0Red";
    env0RedDesc.baseColor = {0.95f, 0.18f, 0.12f};
    env0RedDesc.roughness = 0.4f;
    const auto env0Red = resources.registerMaterial(env0RedDesc);

    MaterialResourceDesc env0OrangeDesc{};
    env0OrangeDesc.debugName = "TwoEnv.Env0Orange";
    env0OrangeDesc.baseColor = {0.95f, 0.58f, 0.12f};
    env0OrangeDesc.roughness = 0.5f;
    const auto env0Orange = resources.registerMaterial(env0OrangeDesc);

    MaterialResourceDesc env1BlueDesc{};
    env1BlueDesc.debugName = "TwoEnv.Env1Blue";
    env1BlueDesc.baseColor = {0.10f, 0.42f, 0.95f};
    env1BlueDesc.roughness = 0.4f;
    const auto env1Blue = resources.registerMaterial(env1BlueDesc);

    MaterialResourceDesc env1GreenDesc{};
    env1GreenDesc.debugName = "TwoEnv.Env1Green";
    env1GreenDesc.baseColor = {0.10f, 0.82f, 0.36f};
    env1GreenDesc.roughness = 0.5f;
    const auto env1Green = resources.registerMaterial(env1GreenDesc);

    cressim::neo::common::EntityId env0Camera = cressim::neo::common::kInvalidEntityId;
    cressim::neo::common::EntityId env1Camera = cressim::neo::common::kInvalidEntityId;

    authorEnvironment(world, resources, 0u, 0.0f, {0.0f, 1.8f, -4.2f}, cubeMesh, planeMesh,
                      env0GroundMaterial, env0Red, env0Orange, env0Camera, true, 0);
    authorEnvironment(world, resources, 1u, 0.5f, {0.0f, 1.8f, -4.2f}, cubeMesh, planeMesh,
                      env1GroundMaterial, env1Blue, env1Green, env1Camera, false, 1);

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = env0Camera;

    const bool runOk = viewer.run(runtime, binding, {});

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        std::cerr << "Viewer run failed.\n";
        return 1;
    }

    std::cout << "Two-env viewer finished.\n";
    return 0;
}
