#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include "graphics/render_resource_manager.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::StrandComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::physics::ColliderShapeType;
using cressim::neo::physics::RigidBodyType;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

} // namespace

int main(int argc, char **argv)
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
    catch (const std::invalid_argument &error)
    {
        CRESSIM_LOG_ERROR(error.what(), "\n");
        printUsage(argv[0]);
        return 2;
    }

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options);

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Strand Debug";
    auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);
    viewerDesc.enableDebugParticles = true;
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

    auto &world = runtime.getWorld();
    auto &resources = runtime.getResources();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.8f, -8.0f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 50.0f;
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.35f, -1.0f, 0.2f};
    light.intensity = 6.0f;
    world.setDirectionalLight(lightEntity, light);

    const auto planeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makePlaneMesh(6.0f, "StrandDebug.PlaneMesh"));
    MaterialResourceDesc groundMaterialDesc{};
    groundMaterialDesc.debugName = "StrandDebug.GroundMaterial";
    groundMaterialDesc.baseColor = {0.72f, 0.75f, 0.79f};
    groundMaterialDesc.metallic  = 0.0f;
    groundMaterialDesc.roughness = 0.92f;
    const auto groundMaterial = resources.registerMaterial(groundMaterialDesc);

    const auto groundEntity = world.createEntity();
    TransformComponent groundTransform{};
    groundTransform.worldTransform.position = {0.0f, -1.2f, 0.0f};
    world.setTransform(groundEntity, groundTransform);
    world.setMeshRenderer(
        groundEntity, cressim::neo::engine::MeshRendererComponent{planeMesh, groundMaterial, true});
    RigidBodyComponent groundBody{};
    groundBody.bodyType    = RigidBodyType::Static;
    groundBody.inverseMass = 0.0f;
    world.setRigidBody(groundEntity, groundBody);
    ColliderComponent groundCollider{};
    groundCollider.shapeType   = ColliderShapeType::Box;
    groundCollider.shapeParams = {6.0f, 0.05f, 6.0f, 0.0f};
    world.addCollider(groundEntity, groundCollider);

    for (std::uint32_t strandIndex = 0u; strandIndex < 3u; ++strandIndex)
    {
        const auto strandEntity = world.createEntity();
        StrandComponent strand{};
        strand.particleMass         = 0.18f;
        strand.particleRadius       = 0.10f;
        strand.distanceCompliance   = 0.0001f;
        strand.bendCompliance       = 0.05f;
        strand.selfCollisionEnabled = false;
        strand.staticParticleIndices = {0u};

        const float baseX = -2.0f + 2.0f * static_cast<float>(strandIndex);
        const float baseY = 2.0f + 0.35f * static_cast<float>(strandIndex);
        const float baseZ = -0.6f + 0.6f * static_cast<float>(strandIndex);
        for (std::uint32_t i = 0u; i < 18u; ++i)
        {
            strand.restPositions.push_back(
                {baseX + 0.25f * static_cast<float>(i), baseY - 0.03f * static_cast<float>(i),
                 baseZ});
        }

        if (!world.setStrand(strandEntity, strand))
        {
            CRESSIM_LOG_ERROR("Failed to author strand debug example entity.\n");
            runtime.shutdown();
            viewer.shutdown();
            return 1;
        }
    }

    DebugViewerCameraBinding binding{};
    binding.cameraEntity = cameraEntity;
    const bool runOk = viewer.run(runtime, binding, {});

    runtime.shutdown();
    viewer.shutdown();

    if (!runOk)
    {
        CRESSIM_LOG_ERROR("Viewer run failed.\n");
        return 1;
    }

    CRESSIM_LOG_INFO("Strand debug example passed.\n");
    return 0;
}
