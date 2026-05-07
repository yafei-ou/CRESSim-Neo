#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{

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
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

void printUsage(const char* appName)
{
    cressim::neo::examples::helpers::printUsage(appName, " [--wave-size N]", false);
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

void spawnWave(cressim::neo::engine::World& world, MeshHandle cubeMesh,
               cressim::neo::graphics::MaterialHandle material, std::uint32_t startIndex,
               std::uint32_t count, bool dynamicBodies)
{
    for (std::uint32_t i = 0u; i < count; ++i)
    {
        const std::uint32_t index = startIndex + i;
        const auto entity = world.createEntity();

        TransformComponent transform{};
        transform.worldTransform.position = {
            static_cast<float>(index % 12u) * 2.1f - 11.5f,
            dynamicBodies ? (5.0f + static_cast<float>((index / 12u) % 3u) * 1.4f)
                          : (0.6f + static_cast<float>((index / 12u) % 6u) * 0.85f),
            dynamicBodies ? (static_cast<float>(index / 36u) * 2.0f)
                          : (8.0f + static_cast<float>(index / 72u) * 2.4f)};
        transform.worldTransform.scale = {1.0f, 1.0f, 1.0f};
        world.setTransform(entity, transform);

        MeshRendererComponent meshRenderer{};
        meshRenderer.mesh = cubeMesh;
        meshRenderer.material = material;
        meshRenderer.visible = true;
        world.setMeshRenderer(entity, meshRenderer);

        RigidBodyComponent body{};
        body.simulated = true;
        body.bodyType = dynamicBodies ? cressim::neo::physics::RigidBodyType::Dynamic
                                      : cressim::neo::physics::RigidBodyType::Static;
        body.inverseMass = dynamicBodies ? 1.0f : 0.0f;
        body.inverseInertiaLocal =
            computeBoxInverseInertia({0.35f, 0.35f, 0.35f}, body.inverseMass);
        body.linearVelocity = {0.0f, 0.0f, 0.0f};
        if (dynamicBodies)
        {
            body.linearVelocity = {static_cast<float>(static_cast<int>(index % 3u) - 1) * 0.05f,
                                   0.0f,
                                   static_cast<float>(static_cast<int>((index / 3u) % 3u) - 1) *
                                       0.05f};
        }
        world.setRigidBody(entity, body);

        ColliderComponent collider{};
        collider.shapeType = ColliderShapeType::Box;
        collider.shapeParams = {0.35f, 0.35f, 0.35f, 0.0f};
        world.addCollider(entity, collider);
    }
}

} // namespace

int main(int argc, char** argv)
{
    CommonExampleOptions options{};
    options.maxFrames = 240u;
    std::uint32_t waveSize = 20u;

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, false))
            {
                continue;
            }

            const std::string arg = argv[i];
            if (arg == "--wave-size")
            {
                waveSize = cressim::neo::examples::helpers::parseEnvCount(
                    cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--wave-size"));
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

    DebugViewerApp viewer;
    const auto viewerDesc = cressim::neo::examples::helpers::makeViewerDesc(
        options, ViewerExampleDefaults{
                     .windowTitle = "CRESSim Neo Physics Viewer Reallocation During Simulation",
                     .width = 1280u,
                     .height = 720u,
                     .showStats = false,
                     .vSync = false,
                     .startFullscreen = false,
                     .startFullscreenWindowed = true,
                 });

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR("Viewer initialization failed.");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Runtime initialization failed.");
        return 1;
    }

    auto& world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 12.0f, -26.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    auto& resources = runtime.getResources();
    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(0.35f, "ViewerRealloc.CubeMesh"));
    const auto planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(18.0f, "ViewerRealloc.PlaneMesh"));

    MaterialResourceDesc dynamicMaterialDesc{};
    dynamicMaterialDesc.debugName = "ViewerRealloc.DynamicMaterial";
    dynamicMaterialDesc.baseColor = {0.90f, 0.45f, 0.12f};
    dynamicMaterialDesc.metallic = 0.0f;
    dynamicMaterialDesc.roughness = 0.55f;
    const auto dynamicMaterial = resources.registerMaterial(dynamicMaterialDesc);

    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "ViewerRealloc.GroundMaterial";
    groundMaterialDesc.baseColor = {0.72f, 0.74f, 0.77f};
    groundMaterialDesc.metallic = 0.0f;
    groundMaterialDesc.roughness = 0.85f;
    const auto groundMaterial = resources.registerMaterial(groundMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent groundRenderer{};
    groundRenderer.mesh = planeMesh;
    groundRenderer.material = groundMaterial;
    groundRenderer.visible = true;
    world.setMeshRenderer(groundEntity, groundRenderer);

    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = cressim::neo::physics::RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);

    ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {18.0f, 0.05f, 18.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    std::uint32_t totalBodies = 0u;
    spawnWave(world, cubeMesh, dynamicMaterial, totalBodies, 4u, true);
    totalBodies += 4u;

    constexpr std::array<std::uint64_t, 3> kWaveFrames = {20u, 80u, 140u};
    std::uint64_t beforeCalls = 0;
    std::uint64_t afterCalls = 0;

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&](const FrameContext& frame, Runtime& cbRuntime) {
        ++beforeCalls;

        for (std::uint32_t waveIndex = 0u; waveIndex < kWaveFrames.size(); ++waveIndex)
        {
            if (frame.frameIndex == kWaveFrames[waveIndex])
            {
                spawnWave(cbRuntime.getWorld(), cubeMesh, dynamicMaterial, totalBodies, waveSize,
                          true);
                totalBodies += waveSize;
            }
        }
    };
    callbacks.afterTick = [&](const FrameContext&, Runtime&) {
        ++afterCalls;
    };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.");
        return 1;
    }
    if (viewerDesc.maxFrames > 0 &&
        (beforeCalls != viewerDesc.maxFrames || afterCalls != viewerDesc.maxFrames))
    {
        CRESSIM_LOG_ERROR("Unexpected callback counts. before=", beforeCalls, " after=", afterCalls,
                          " expected=", viewerDesc.maxFrames);
        return 1;
    }

    CRESSIM_LOG_INFO("Physics viewer reallocation-during-simulation passed. Frames=",
                     viewerDesc.maxFrames, " totalBodies=", totalBodies);
    return 0;
}
