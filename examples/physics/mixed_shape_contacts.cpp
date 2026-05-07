#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/inertia.h"
#include "helpers/shape_meshes.h"
#include "viewer/debug_viewer_app.h"
#include "common/logger.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
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
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

constexpr float kCompositeChildHalfExtent = 0.60f;
constexpr float kCompositeHalfExtent = 1.0f;
constexpr float kCompositeChildOffset = kCompositeHalfExtent - kCompositeChildHalfExtent;

GpuBackend parseBackend(const std::string& value)
{
    if (value == "null")
    {
        return GpuBackend::Null;
    }
    if (value == "d3d12")
    {
        return GpuBackend::D3D12;
    }
    if (value == "vulkan")
    {
        return GpuBackend::Vulkan;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

void printUsage(const char* appName)
{
    CRESSIM_LOG_ERROR( "Usage: " , appName , " [--backend vulkan|d3d12|null] [--frames N]\n");
}

Diligent::float4 colliderParamsForShape(ColliderShapeType shape)
{
    switch (shape)
    {
        case ColliderShapeType::Sphere:
            return {0.6f, 0.0f, 0.0f, 0.0f};
        case ColliderShapeType::Box:
            return {0.55f, 0.45f, 0.5f, 0.0f};
        case ColliderShapeType::Capsule:
            return {0.35f, 0.55f, 0.0f, 0.0f};
    }

    return {0.55f, 0.45f, 0.5f, 0.0f};
}

struct BodySpawn
{
    ColliderShapeType shape = ColliderShapeType::Box;
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    cressim::neo::graphics::MaterialHandle material{};
};

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
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

    DebugViewerApp viewer;
    DebugViewerAppDesc viewerDesc{};
    const bool windowEnabled = (config.gpuDeviceDesc.preferredBackend != GpuBackend::Null);
    viewerDesc.windowEnabled = windowEnabled;
    viewerDesc.windowVisible = windowEnabled;
    viewerDesc.startFullscreenWindowed = true;
    viewerDesc.maxFrames = numFrames;
    viewerDesc.showStats = false;
    viewerDesc.width = 960;
    viewerDesc.height = 540;

    if (!viewer.initialize(viewerDesc, config))
    {
        CRESSIM_LOG_ERROR( "Viewer initialization failed.\n");
        return 1;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        viewer.shutdown();
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    auto& world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.6f, -6.2f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 52.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.45f, -1.0f, 0.35f};
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    auto& resources = runtime.getResources();
    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(0.5f, "MixedShapeViewer.CubeMesh"));
    const auto largeCubeMesh = resources.registerMesh(cressim::neo::examples::helpers::makeCubeMesh(
        kCompositeHalfExtent, "MixedShapeViewer.CubeMesh"));
    const auto planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(8.0f, "MixedShapeViewer.PlaneMesh"));

    MaterialResourceDesc compositeMaterialDesc{};
    compositeMaterialDesc.debugName = "MixedShapeViewer.CompositeBodyMaterial";
    compositeMaterialDesc.baseColor = {0.16f, 0.62f, 0.96f};
    compositeMaterialDesc.metallic = 0.0f;
    compositeMaterialDesc.roughness = 0.40f;
    const auto compositeMaterial = resources.registerMaterial(compositeMaterialDesc);

    MaterialResourceDesc probeMaterialDesc{};
    probeMaterialDesc.debugName = "MixedShapeViewer.ProbeBodyMaterial";
    probeMaterialDesc.baseColor = {0.92f, 0.26f, 0.18f};
    probeMaterialDesc.metallic = 0.0f;
    probeMaterialDesc.roughness = 0.42f;
    const auto probeMaterial = resources.registerMaterial(probeMaterialDesc);

    MaterialResourceDesc planeMaterialDesc{};
    planeMaterialDesc.debugName = "MixedShapeViewer.GroundMaterial";
    planeMaterialDesc.baseColor = {0.72f, 0.74f, 0.77f};
    planeMaterialDesc.metallic = 0.0f;
    planeMaterialDesc.roughness = 0.85f;
    const auto planeMaterial = resources.registerMaterial(planeMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent groundMesh{};
    groundMesh.mesh = planeMesh;
    groundMesh.material = planeMaterial;
    groundMesh.visible = true;
    world.setMeshRenderer(groundEntity, groundMesh);
    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = cressim::neo::physics::RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType = ColliderShapeType::Box;
    groundCollider.shapeParams = {8.0f, 0.05f, 8.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    const auto compositeEntity = world.createEntity();
    TransformComponent compositeTransform{};
    compositeTransform.worldTransform.position = {0.0f, 1.85f, 2.0f};
    world.setTransform(compositeEntity, compositeTransform);

    MeshRendererComponent compositeMesh{};
    compositeMesh.mesh = largeCubeMesh;
    compositeMesh.material = compositeMaterial;
    compositeMesh.visible = true;
    world.setMeshRenderer(compositeEntity, compositeMesh);

    RigidBodyComponent compositeBody{};
    compositeBody.simulated = true;
    compositeBody.inverseMass = 1.0f;
    compositeBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(
            {kCompositeHalfExtent, kCompositeHalfExtent, kCompositeHalfExtent},
            compositeBody.inverseMass);
    world.setRigidBody(compositeEntity, compositeBody);

    const std::vector<Diligent::float3> colliderOffsets = {
        {-kCompositeChildOffset, -kCompositeChildOffset, -kCompositeChildOffset},
        {kCompositeChildOffset, -kCompositeChildOffset, -kCompositeChildOffset},
        {-kCompositeChildOffset, kCompositeChildOffset, -kCompositeChildOffset},
        {kCompositeChildOffset, kCompositeChildOffset, -kCompositeChildOffset},
        {-kCompositeChildOffset, -kCompositeChildOffset, kCompositeChildOffset},
        {kCompositeChildOffset, -kCompositeChildOffset, kCompositeChildOffset},
        {-kCompositeChildOffset, kCompositeChildOffset, kCompositeChildOffset},
        {kCompositeChildOffset, kCompositeChildOffset, kCompositeChildOffset},
    };

    for (const Diligent::float3& offset : colliderOffsets)
    {
        ColliderComponent collider{};
        collider.shapeType = ColliderShapeType::Box;
        collider.shapeParams = {kCompositeChildHalfExtent, kCompositeChildHalfExtent,
                                kCompositeChildHalfExtent, 0.0f};
        collider.localPosition = offset;
        world.addCollider(compositeEntity, collider);
    }

    const auto singleColliderEntity = world.createEntity();
    TransformComponent singleColliderTransform{};
    singleColliderTransform.worldTransform.position = {3.6f, 1.85f, 2.0f};
    world.setTransform(singleColliderEntity, singleColliderTransform);

    MeshRendererComponent singleColliderMesh{};
    singleColliderMesh.mesh = largeCubeMesh;
    singleColliderMesh.material = probeMaterial;
    singleColliderMesh.visible = true;
    world.setMeshRenderer(singleColliderEntity, singleColliderMesh);

    RigidBodyComponent singleColliderBody{};
    singleColliderBody.simulated = true;
    singleColliderBody.inverseMass = 1.0f;
    singleColliderBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(
            {kCompositeHalfExtent, kCompositeHalfExtent, kCompositeHalfExtent},
            singleColliderBody.inverseMass);
    world.setRigidBody(singleColliderEntity, singleColliderBody);

    ColliderComponent singleCollider{};
    singleCollider.shapeType = ColliderShapeType::Box;
    singleCollider.shapeParams = {kCompositeHalfExtent, kCompositeHalfExtent,
                                  kCompositeHalfExtent, 0.0f};
    world.addCollider(singleColliderEntity, singleCollider);

    // const auto probeEntity = world.createEntity();
    // TransformComponent probeTransform{};
    // probeTransform.worldTransform.position = {1.55f, 1.15f, 2.0f};
    // probeTransform.worldTransform.scale = {0.9f, 0.9f, 0.9f};
    // world.setTransform(probeEntity, probeTransform);

    // MeshRendererComponent probeMesh{};
    // probeMesh.mesh = cubeMesh;
    // probeMesh.material = probeMaterial;
    // probeMesh.visible = true;
    // world.setMeshRenderer(probeEntity, probeMesh);

    // RigidBodyComponent probeBody{};
    // probeBody.simulated = true;
    // probeBody.inverseMass = 1.0f;
    // probeBody.inverseInertiaLocal = computeBoxInverseInertia({0.45f, 0.45f, 0.45f},
    //                                                          probeBody.inverseMass);
    // world.setRigidBody(probeEntity, probeBody);

    // ColliderComponent probeCollider{};
    // probeCollider.shapeType = ColliderShapeType::Box;
    // probeCollider.shapeParams = {0.45f, 0.45f, 0.45f, 0.0f};
    // world.addCollider(probeEntity, probeCollider);

    std::uint64_t beforeCalls = 0;
    std::uint64_t afterCalls = 0;

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [&](const FrameContext&, Runtime&) { ++beforeCalls; };
    callbacks.afterTick = [&](const FrameContext&, Runtime&) { ++afterCalls; };

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, callbacks);

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR( "Viewer run failed.\n");
        return 1;
    }
    if (viewerDesc.maxFrames > 0 &&
        (beforeCalls != viewerDesc.maxFrames || afterCalls != viewerDesc.maxFrames))
    {
        CRESSIM_LOG_ERROR( "Unexpected callback counts. before=" , beforeCalls , " after=" , afterCalls
                  , " expected=" , viewerDesc.maxFrames , '\n');
        return 1;
    }

    CRESSIM_LOG_INFO( "Physics mixed-shape viewer passed. Frames=" , viewerDesc.maxFrames , '\n');
    return 0;
}
