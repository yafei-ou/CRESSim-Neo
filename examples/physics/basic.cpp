#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/inertia.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCallbacks;
using cressim::neo::viewer::DebugViewerCameraBinding;

void printUsage(const char* appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

} // namespace

int main(int argc, char** argv)
{
    CommonExampleOptions options{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options, false))
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

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Physics Viewer";
    const auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);

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

    const auto frontCubeEntity = world.createEntity();
    TransformComponent frontCubeTransform{};
    frontCubeTransform.worldTransform.position = {0.65f, 1.25f, 2.25f};
    world.setTransform(frontCubeEntity, frontCubeTransform);
    MeshRendererComponent frontCube{};
    frontCube.mesh = cubeMesh;
    frontCube.material = frontMaterial;
    frontCube.visible = true;
    world.setMeshRenderer(frontCubeEntity, frontCube);
    RigidBodyComponent frontCubeBody{};
    frontCubeBody.simulated = true;
    frontCubeBody.inverseMass = 1.0f;
    frontCubeBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(
            {0.65f, 0.65f, 0.65f}, frontCubeBody.inverseMass);
    frontCubeBody.linearVelocity = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(frontCubeEntity, frontCubeBody);
    cressim::neo::engine::ColliderComponent frontCubeCollider{};
    frontCubeCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
    frontCubeCollider.shapeParams = {0.65f, 0.65f, 0.65f, 0.0f};
    world.addCollider(frontCubeEntity, frontCubeCollider);

    const auto backCubeEntity = world.createEntity();
    TransformComponent backCubeTransform{};
    backCubeTransform.worldTransform.position = {1.25f, 3.35f, 2.15f};
    backCubeTransform.worldTransform.scale = {1.15f, 1.15f, 1.15f};
    world.setTransform(backCubeEntity, backCubeTransform);
    MeshRendererComponent backCube{};
    backCube.mesh = cubeMesh;
    backCube.material = backMaterial;
    backCube.visible = true;
    world.setMeshRenderer(backCubeEntity, backCube);
    RigidBodyComponent backCubeBody{};
    backCubeBody.simulated = true;
    backCubeBody.inverseMass = 1.0f;
    backCubeBody.inverseInertiaLocal =
        cressim::neo::examples::helpers::computeBoxInverseInertia(
            {0.65f * 1.15f, 0.65f * 1.15f, 0.65f * 1.15f}, backCubeBody.inverseMass);
    backCubeBody.linearVelocity = {0.0f, 0.0f, 0.0f};
    world.setRigidBody(backCubeEntity, backCubeBody);
    cressim::neo::engine::ColliderComponent backCubeCollider{};
    backCubeCollider.shapeType = cressim::neo::physics::ColliderShapeType::Box;
    backCubeCollider.shapeParams = {0.65f, 0.65f, 0.65f, 0.0f};
    world.addCollider(backCubeEntity, backCubeCollider);

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
