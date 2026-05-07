#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/shape_meshes.h"
#include "viewer/debug_viewer_app.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerAppDesc;
using cressim::neo::viewer::DebugViewerCallbacks;
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

ColliderShapeType parseColliderShape(const std::string& value)
{
    if (value == "sphere")
    {
        return ColliderShapeType::Sphere;
    }
    if (value == "box")
    {
        return ColliderShapeType::Box;
    }
    if (value == "capsule")
    {
        return ColliderShapeType::Capsule;
    }
    throw std::invalid_argument("Unsupported shape: " + value);
}

bool parseShapePair(const std::string& value, ColliderShapeType& outA,
                    ColliderShapeType& outB)
{
    const std::size_t split = value.find('-');
    if (split == std::string::npos || split == 0u || split + 1u >= value.size())
    {
        return false;
    }

    const std::string a = value.substr(0u, split);
    const std::string b = value.substr(split + 1u);
    outA = parseColliderShape(a);
    outB = parseColliderShape(b);
    return true;
}

void printUsage(const char* appName)
{
    CRESSIM_LOG_ERROR( "Usage: " , appName
              , " [--backend vulkan|null] [--frames N] [--pair A-B]\n"
              , "  Shapes: box, sphere, capsule\n"
              , "  Unique pairs: box-box, box-sphere, box-capsule, sphere-sphere,\n"
              , "                sphere-capsule, capsule-capsule\n");
}

Diligent::float3 computeBoxInverseInertia(const Diligent::float3& halfExtents, float inverseMass)
{
    if (inverseMass <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float ix = mass * (halfExtents.y * halfExtents.y + halfExtents.z * halfExtents.z) / 3.0f;
    const float iy = mass * (halfExtents.x * halfExtents.x + halfExtents.z * halfExtents.z) / 3.0f;
    const float iz = mass * (halfExtents.x * halfExtents.x + halfExtents.y * halfExtents.y) / 3.0f;

    return {ix > 0.0f ? 1.0f / ix : 0.0f,
            iy > 0.0f ? 1.0f / iy : 0.0f,
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
            return {0.65f, 0.0f, 0.0f, 0.0f};
        case ColliderShapeType::Box:
            return {0.65f, 0.65f, 0.65f, 0.0f};
        case ColliderShapeType::Capsule:
            return {0.40f, 0.60f, 0.0f, 0.0f};
    }

    return {0.65f, 0.65f, 0.65f, 0.0f};
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

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    std::uint64_t numFrames = 0;
    ColliderShapeType shapeA = ColliderShapeType::Box;
    ColliderShapeType shapeB = ColliderShapeType::Box;

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
        if (arg == "--pair")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }

            try
            {
                if (!parseShapePair(argv[++i], shapeA, shapeB))
                {
                    printUsage(argv[0]);
                    return 2;
                }
            }
            catch (const std::invalid_argument&)
            {
                printUsage(argv[0]);
                return 2;
            }
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
    viewerDesc.width = 640;
    viewerDesc.height = 480;

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
    cameraTransform.worldTransform.position = {0.0f, 1.8f, -4.2f};
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
        cressim::neo::examples::helpers::makeCubeMesh(0.65f, "ViewerIntegration.CubeMesh"));
    const auto planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(8.0f, "ViewerIntegration.PlaneMesh"));
    const auto sphereMesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        0.65f, 20u, 12u, "ViewerIntegration.SphereMesh"));
    const auto capsuleMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCapsuleMesh(
            0.40f, 0.60f, 20u, 6u, 2u, "ViewerIntegration.CapsuleMesh"));

    MaterialResourceDesc frontMaterialDesc{};
    frontMaterialDesc.debugName = "ViewerIntegration.FrontMaterial";
    frontMaterialDesc.baseColor = {0.95f, 0.10f, 0.08f};
    frontMaterialDesc.metallic = 0.0f;
    frontMaterialDesc.roughness = 0.45f;
    const auto frontMaterial = resources.registerMaterial(frontMaterialDesc);

    MaterialResourceDesc backMaterialDesc{};
    backMaterialDesc.debugName = "ViewerIntegration.BackMaterial";
    backMaterialDesc.baseColor = {0.10f, 0.85f, 0.12f};
    backMaterialDesc.metallic = 0.0f;
    backMaterialDesc.roughness = 0.45f;
    const auto backMaterial = resources.registerMaterial(backMaterialDesc);

    MaterialResourceDesc planeMaterialDesc{};
    planeMaterialDesc.debugName = "ViewerIntegration.GroundMaterial";
    planeMaterialDesc.baseColor = {0.72f, 0.74f, 0.77f};
    planeMaterialDesc.metallic = 0.0f;
    planeMaterialDesc.roughness = 0.85f;
    const auto planeMaterial = resources.registerMaterial(planeMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.0f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    MeshRendererComponent ground{};
    ground.mesh = planeMesh;
    ground.material = planeMaterial;
    ground.visible = true;
    world.setMeshRenderer(groundEntity, ground);
    RigidBodyComponent groundBody{};
    groundBody.simulated = true;
    groundBody.bodyType = cressim::neo::physics::RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    groundBody.inverseInertiaLocal = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(groundEntity, groundBody);
    cressim::neo::engine::ColliderComponent groundCollider{};
    groundCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
    groundCollider.shapeParams = {8.0f, 0.05f, 8.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

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

    const bool legacyBoxPair = (shapeA == ColliderShapeType::Box && shapeB == ColliderShapeType::Box);
    const Diligent::float3 frontPosition = {0.65f, 1.25f, 2.25f};
    const Diligent::float3 backPosition = {1.25f, 3.35f, 2.15f};
    const Diligent::float3 backScale = legacyBoxPair
                                           ? Diligent::float3{1.15f, 1.15f, 1.15f}
                                           : Diligent::float3{1.0f, 1.0f, 1.0f};

    const auto frontEntity = world.createEntity();
    TransformComponent frontTransform{};
    frontTransform.worldTransform.position = frontPosition;
    world.setTransform(frontEntity, frontTransform);
    MeshRendererComponent frontMesh{};
    frontMesh.mesh = meshForShape(shapeA);
    frontMesh.material = frontMaterial;
    frontMesh.visible = true;
    world.setMeshRenderer(frontEntity, frontMesh);
    RigidBodyComponent frontBody{};
    frontBody.simulated = true;
    frontBody.inverseMass = 1.0f;
    frontBody.inverseInertiaLocal =
        inverseInertiaForShape(shapeA, colliderParamsForShape(shapeA), frontBody.inverseMass);
    frontBody.linearVelocity = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(frontEntity, frontBody);
    cressim::neo::engine::ColliderComponent frontCollider{};
    frontCollider.shapeType = shapeA;
    frontCollider.shapeParams = colliderParamsForShape(shapeA);
    world.addCollider(frontEntity, frontCollider);

    const auto backEntity = world.createEntity();
    TransformComponent backTransform{};
    backTransform.worldTransform.position = backPosition;
    backTransform.worldTransform.scale = backScale;
    world.setTransform(backEntity, backTransform);
    MeshRendererComponent backMesh{};
    backMesh.mesh = meshForShape(shapeB);
    backMesh.material = backMaterial;
    backMesh.visible = true;
    world.setMeshRenderer(backEntity, backMesh);
    RigidBodyComponent backBody{};
    backBody.simulated = true;
    backBody.inverseMass = 1.0f;
    if (legacyBoxPair)
    {
        backBody.inverseInertiaLocal = computeBoxInverseInertia({0.65f * 1.15f, 0.65f * 1.15f,
                                                                 0.65f * 1.15f},
                                                                backBody.inverseMass);
    }
    else
    {
        backBody.inverseInertiaLocal =
            inverseInertiaForShape(shapeB, colliderParamsForShape(shapeB), backBody.inverseMass);
    }
    backBody.linearVelocity = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(backEntity, backBody);
    cressim::neo::engine::ColliderComponent backCollider{};
    backCollider.shapeType = shapeB;
    backCollider.shapeParams = colliderParamsForShape(shapeB);
    world.addCollider(backEntity, backCollider);

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

    CRESSIM_LOG_INFO( "Physics viewer passed. Frames=" , viewerDesc.maxFrames , '\n');
    return 0;
}
