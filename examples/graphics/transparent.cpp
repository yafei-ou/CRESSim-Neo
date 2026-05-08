#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <stdexcept>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MaterialHandle;
using cressim::neo::graphics::MaterialRenderMode;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshHandle;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

MaterialHandle registerMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                const char *name, const Diligent::float3 &baseColor,
                                float metallic, float roughness)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.metallic = metallic;
    desc.roughness = roughness;
    return resources.registerMaterial(desc);
}

MaterialHandle registerTransparentMaterial(cressim::neo::graphics::RenderResourceManager &resources,
                                           const char *name,
                                           const Diligent::float3 &baseColor, float opacity,
                                           float roughness, std::int32_t renderOrder)
{
    MaterialResourceDesc desc{};
    desc.debugName = name;
    desc.baseColor = baseColor;
    desc.opacity = opacity;
    desc.roughness = roughness;
    desc.emissiveFactor = {baseColor.x * 0.10f, baseColor.y * 0.10f, baseColor.z * 0.10f};
    desc.renderMode = MaterialRenderMode::Transparent;
    desc.renderOrder = renderOrder;
    desc.castsShadows = false;
    desc.pipeline.featureFlags = MaterialFeatureFlags::DoubleSided;
    return resources.registerMaterial(desc);
}

void spawnRenderable(cressim::neo::engine::World &world, MeshHandle mesh, MaterialHandle material,
                     const Diligent::float3 &position, const Diligent::float3 &scale,
                     const Diligent::QuaternionF &rotation = Diligent::QuaternionF{})
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale = scale;
    transform.worldTransform.rotation = rotation;
    world.setTransform(entity, transform);

    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(entity, renderer);
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
    config.sceneLayout.envCount = 1u;
    config.sceneLayout.maxRenderableObjectsPerEnv = 16u;
    config.sceneLayout.maxLightsPerEnv = 2u;
    config.sceneLayout.maxCamerasPerEnv = 1u;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Transparent Pass";
    viewerDefaults.showStats = true;
    viewerDefaults.vSync = true;
    const auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options, viewerDefaults);

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

    try
    {
        auto &world = runtime.getWorld();
        auto &resources = runtime.getResources();

        const auto cameraEntity = world.createEntity();
        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {0.0f, 1.9f, -8.0f};
        world.setTransform(cameraEntity, cameraTransform);
        world.setCamera(cameraEntity, CameraComponent{});

        const auto lightEntity = world.createEntity();
        DirectionalLightComponent light{};
        light.direction = {-0.45f, -1.0f, 0.2f};
        light.color = {1.0f, 0.98f, 0.96f};
        light.intensity = 8.0f;
        world.setDirectionalLight(lightEntity, light);

        const auto groundMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makePlaneMesh(10.0f, "TransparentScene.Ground"));
        const auto panelMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makePlaneMesh(1.0f, "TransparentScene.Panel"));
        const auto cubeMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeCubeMesh(0.7f, "TransparentScene.Cube"));
        const auto sphereMesh = resources.registerMesh(
            cressim::neo::examples::helpers::makeSphereMesh(0.8f, 40u, 20u,
                                                            "TransparentScene.Sphere"));

        const auto groundMaterial =
            registerMaterial(resources, "TransparentScene.GroundMat", {0.24f, 0.25f, 0.28f}, 0.0f,
                             0.96f);
        const auto wallMaterial =
            registerMaterial(resources, "TransparentScene.WallMat", {0.16f, 0.17f, 0.20f}, 0.0f,
                             0.75f);
        const auto redCubeMaterial =
            registerMaterial(resources, "TransparentScene.RedCube", {0.86f, 0.22f, 0.20f}, 0.0f,
                             0.42f);
        const auto brassSphereMaterial = registerMaterial(resources, "TransparentScene.BrassSphere",
                                                          {0.98f, 0.82f, 0.34f}, 1.0f, 0.24f);
        const auto cyanGlassMaterial = registerTransparentMaterial(
            resources, "TransparentScene.CyanGlass", {0.10f, 0.88f, 0.98f}, 0.62f, 0.52f,
            0);
        const auto amberGlassMaterial = registerTransparentMaterial(
            resources, "TransparentScene.AmberGlass", {0.98f, 0.58f, 0.10f}, 0.56f, 0.48f,
            10);
        const auto magentaGlassMaterial = registerTransparentMaterial(
            resources, "TransparentScene.MagentaGlass", {0.96f, 0.14f, 0.74f}, 0.50f, 0.44f,
            20);

        const Diligent::QuaternionF floorAligned =
            Diligent::QuaternionF::RotationFromAxisAngle({1.0f, 0.0f, 0.0f},
                                                         -Diligent::PI_F * 0.5f);
        const Diligent::QuaternionF frontPanelRotation =
            Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, 0.18f);
        const Diligent::QuaternionF rearPanelRotation =
            Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, -0.34f);
        const Diligent::QuaternionF sidePanelRotation =
            Diligent::QuaternionF::RotationFromAxisAngle({0.0f, 1.0f, 0.0f}, Diligent::PI_F * 0.5f);

        spawnRenderable(world, groundMesh, groundMaterial, {0.0f, -1.4f, 4.5f},
                        {1.0f, 1.0f, 1.0f});
        spawnRenderable(world, panelMesh, wallMaterial, {0.0f, 1.4f, 8.0f},
                        {8.0f, 1.0f, 3.0f}, floorAligned);
        spawnRenderable(world, panelMesh, wallMaterial, {-3.8f, 1.2f, 5.0f},
                        {1.2f, 1.0f, 2.1f}, floorAligned * frontPanelRotation);
        spawnRenderable(world, panelMesh, wallMaterial, {3.9f, 1.2f, 5.4f},
                        {1.4f, 1.0f, 2.0f}, floorAligned * rearPanelRotation);

        spawnRenderable(world, cubeMesh, redCubeMaterial, {-1.8f, -0.5f, 4.2f},
                        {1.0f, 1.0f, 1.0f});
        spawnRenderable(world, sphereMesh, brassSphereMaterial, {1.6f, -0.55f, 5.1f},
                        {1.0f, 1.0f, 1.0f});
        spawnRenderable(world, cubeMesh, redCubeMaterial, {0.0f, 0.2f, 6.5f},
                        {0.65f, 1.4f, 0.65f});

        spawnRenderable(world, panelMesh, cyanGlassMaterial, {-0.8f, 1.0f, 3.9f},
                        {2.0f, 1.0f, 2.4f}, floorAligned * frontPanelRotation);
        spawnRenderable(world, panelMesh, amberGlassMaterial, {0.7f, 1.05f, 4.8f},
                        {2.0f, 1.0f, 2.2f}, floorAligned * rearPanelRotation);
        spawnRenderable(world, panelMesh, magentaGlassMaterial, {2.8f, 1.0f, 4.6f},
                        {1.5f, 1.0f, 2.0f}, floorAligned * sidePanelRotation);

        DebugViewerCameraBinding binding{};
        binding.cameraEntity = cameraEntity;
        const bool runOk = viewer.run(runtime, binding, {});

        runtime.shutdown();
        viewer.shutdown();

        if (!runOk)
        {
            CRESSIM_LOG_ERROR("Transparent pass viewer run failed.\n");
            return 1;
        }
    }
    catch (const std::exception &ex)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Transparent scene setup failed: ", ex.what(), '\n');
        return 1;
    }

    CRESSIM_LOG_INFO(
        "Transparent viewer finished. Visual checks: cyan/amber/magenta panels should show distinct tinting, and higher transparent render orders should appear on top of lower ones where they overlap.\n");
    return 0;
}
