#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <array>
#include <cstdint>
#include <string>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr std::uint32_t kDefaultEnvCount = 1u;

struct ViewerPalette
{
    Diligent::float3 ground{0.72f, 0.74f, 0.77f};
    Diligent::float3 first{0.95f, 0.18f, 0.12f};
    Diligent::float3 second{0.95f, 0.58f, 0.12f};
};

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "DebugViewer.CubeMesh";
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
    mesh.debugName = "DebugViewer.PlaneMesh";
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

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager& resources,
                                const std::string& debugName,
                                const Diligent::float3& baseColor, float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = debugName;
    desc.baseColor = baseColor;
    desc.metallic = 0.0f;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

ViewerPalette paletteForEnv(std::uint32_t envIndex)
{
    static constexpr std::array<ViewerPalette, 6> kPalettes = {{
        {{0.75f, 0.73f, 0.72f}, {0.95f, 0.18f, 0.12f}, {0.95f, 0.58f, 0.12f}},
        {{0.70f, 0.78f, 0.92f}, {0.10f, 0.42f, 0.95f}, {0.10f, 0.82f, 0.36f}},
        {{0.78f, 0.74f, 0.68f}, {0.92f, 0.26f, 0.52f}, {0.22f, 0.70f, 0.94f}},
        {{0.68f, 0.74f, 0.86f}, {0.26f, 0.46f, 0.94f}, {0.94f, 0.30f, 0.24f}},
        {{0.74f, 0.71f, 0.64f}, {0.88f, 0.42f, 0.16f}, {0.20f, 0.76f, 0.66f}},
        {{0.70f, 0.78f, 0.70f}, {0.72f, 0.20f, 0.20f}, {0.20f, 0.76f, 0.88f}},
    }};
    return kPalettes[envIndex % kPalettes.size()];
}

void authorEnvironment(cressim::neo::engine::World& world, std::uint32_t envIndex,
                       MeshHandle cubeMesh, MeshHandle planeMesh, MaterialHandle groundMaterial,
                       MaterialHandle cubeMaterialA, MaterialHandle cubeMaterialB,
                       EntityId& outCameraEntity)
{
    outCameraEntity = world.createEntity(envIndex);

    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 1.8f, -4.2f};
    world.setTransform(outCameraEntity, cameraTransform);

    CameraComponent camera{};
    camera.clearColor = true;
    camera.clearDepth = true;
    camera.renderOrder = envIndex;
    world.setCamera(outCameraEntity, camera);

    const auto lightEntity = world.createEntity(envIndex);
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    const auto groundEntity = world.createEntity(envIndex);
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(groundEntity, MeshRendererComponent{planeMesh, groundMaterial, true});

    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = cressim::neo::physics::RigidBodyType::Static;
    world.setRigidBody(groundEntity, groundBody);

    ColliderComponent groundCollider{};
    groundCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
    groundCollider.shapeParams = {8.0f, 0.05f, 8.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    const bool firstCubeIsRigidBody = (envIndex % 2u == 0u);

    const auto cubeAEntity = world.createEntity(envIndex);
    TransformComponent cubeATransform{};
    cubeATransform.worldTransform.position = {-0.8f, 0.3f, 0.2f};
    world.setTransform(cubeAEntity, cubeATransform);
    world.setMeshRenderer(cubeAEntity,
                          MeshRendererComponent{cubeMesh, cubeMaterialA, true});
    if (firstCubeIsRigidBody)
    {
        RigidBodyComponent cubeBody{};
        cubeBody.simulated = true;
        cubeBody.inverseMass = 1.0f;
        cubeBody.inverseInertiaLocal =
            computeBoxInverseInertia({0.65f, 0.65f, 0.65f}, cubeBody.inverseMass);
        world.setRigidBody(cubeAEntity, cubeBody);

        ColliderComponent cubeCollider{};
        cubeCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
        cubeCollider.shapeParams = {0.65f, 0.65f, 0.65f, 0.0f};
        world.addCollider(cubeAEntity, cubeCollider);
    }

    const auto cubeBEntity = world.createEntity(envIndex);
    TransformComponent cubeBTransform{};
    cubeBTransform.worldTransform.position = {1.0f, 0.85f, 1.3f};
    cubeBTransform.worldTransform.scale = {1.15f, 1.15f, 1.15f};
    world.setTransform(cubeBEntity, cubeBTransform);
    world.setMeshRenderer(cubeBEntity,
                          MeshRendererComponent{cubeMesh, cubeMaterialB, true});
    if (!firstCubeIsRigidBody)
    {
        RigidBodyComponent cubeBody{};
        cubeBody.simulated = true;
        cubeBody.inverseMass = 1.0f;
        cubeBody.inverseInertiaLocal = computeBoxInverseInertia(
            {0.65f * 1.15f, 0.65f * 1.15f, 0.65f * 1.15f}, cubeBody.inverseMass);
        world.setRigidBody(cubeBEntity, cubeBody);

        ColliderComponent cubeCollider{};
        cubeCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
        cubeCollider.shapeParams = {0.65f * 1.15f, 0.65f * 1.15f, 0.65f * 1.15f, 0.0f};
        world.addCollider(cubeBEntity, cubeCollider);
    }
}

void printUsage(const char* appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", true);
}

} // namespace

int main(int argc, char** argv)
{
    CommonExampleOptions options{};
    options.envCount = kDefaultEnvCount;

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, true))
            {
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

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);
    config.sceneLayout.envCount = options.envCount;
    config.sceneLayout.maxRenderableObjectsPerEnv = 8u;
    config.sceneLayout.maxLightsPerEnv = 2u;
    config.sceneLayout.maxCamerasPerEnv = 2u;

    DebugViewerApp viewer;
    const auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(
        options, ViewerExampleDefaults{
                     .windowTitle = "CRESSim Neo Debug Viewer",
                     .width = 1280u,
                     .height = 720u,
                     .showStats = false,
                     .vSync = false,
                     .startFullscreen = false,
                     .startFullscreenWindowed = true,
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

    const auto cubeMesh = resources.registerMesh(makeCubeMesh(0.65f));
    const auto planeMesh = resources.registerMesh(makePlaneMesh(8.0f));

    EntityId presentedCamera = cressim::neo::common::kInvalidEntityId;
    for (std::uint32_t envIndex = 0u; envIndex < options.envCount; ++envIndex)
    {
        const auto palette = paletteForEnv(envIndex);
        const auto groundMaterial = registerMaterial(
            resources, "DebugViewer.Ground." + std::to_string(envIndex), palette.ground, 0.9f);
        const auto firstMaterial = registerMaterial(
            resources, "DebugViewer.First." + std::to_string(envIndex), palette.first, 0.4f);
        const auto secondMaterial = registerMaterial(
            resources, "DebugViewer.Second." + std::to_string(envIndex), palette.second, 0.5f);

        EntityId cameraEntity = cressim::neo::common::kInvalidEntityId;
        authorEnvironment(world, envIndex, cubeMesh, planeMesh, groundMaterial, firstMaterial,
                          secondMaterial, cameraEntity);
        if (envIndex == 0u)
        {
            presentedCamera = cameraEntity;
        }
    }

    std::uint64_t beforeCalls = 0u;
    std::uint64_t afterCalls = 0u;

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&](const FrameContext&, Runtime&) { ++beforeCalls; };
    callbacks.afterTick = [&](const FrameContext&, Runtime&) { ++afterCalls; };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = presentedCamera;
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

    return 0;
}
