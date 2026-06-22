#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <cstdint>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{

using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::TransformComponent;
using cressim::neo::examples::helpers::CommonExampleOptions;
using cressim::neo::examples::helpers::ViewerExampleDefaults;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::viewer::DebugViewerApp;
using cressim::neo::viewer::DebugViewerCameraBinding;
using cressim::neo::viewer::DebugViewerCallbacks;

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(appName, "", false);
}

Diligent::QuaternionF quaternionFromEulerDegrees(float pitchDegrees, float yawDegrees,
                                                 float rollDegrees)
{
    const float pitch = pitchDegrees * 0.017453292519943295769f * 0.5f;
    const float yaw   = yawDegrees * 0.017453292519943295769f * 0.5f;
    const float roll  = rollDegrees * 0.017453292519943295769f * 0.5f;

    const float sinPitch = std::sin(pitch);
    const float cosPitch = std::cos(pitch);
    const float sinYaw   = std::sin(yaw);
    const float cosYaw   = std::cos(yaw);
    const float sinRoll  = std::sin(roll);
    const float cosRoll  = std::cos(roll);

    return Diligent::QuaternionF{
        sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
        cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
        cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
        cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw};
}

void spawnRenderable(cressim::neo::engine::World &world, cressim::neo::graphics::MeshHandle mesh,
                     cressim::neo::graphics::MaterialHandle material,
                     const Diligent::float3 &position, const Diligent::float3 &scale,
                     const Diligent::QuaternionF &rotation = Diligent::QuaternionF{})
{
    const auto entity = world.createEntity();
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale    = scale;
    transform.worldTransform.rotation = rotation;
    world.setTransform(entity, transform);

    MeshRendererComponent renderer{};
    renderer.mesh     = mesh;
    renderer.material = material;
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

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Depth Camera Example";
    viewerDefaults.showStats   = true;
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

    auto *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Graphics device unavailable.\n");
        return 1;
    }

    auto &world = runtime.getWorld();

    const auto colorCameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 2.25f, -8.0f};
    world.setTransform(colorCameraEntity, cameraTransform);
    CameraComponent colorCamera{};
    colorCamera.verticalFovDegrees = 50.0f;
    colorCamera.nearClip           = 0.05f;
    colorCamera.farClip            = 40.0f;
    world.setCamera(colorCameraEntity, colorCamera);

    cressim::neo::gpu::GpuRenderTargetDesc depthTargetDesc{};
    depthTargetDesc.width     = 960u;
    depthTargetDesc.height    = 540u;
    depthTargetDesc.color     = false;
    depthTargetDesc.depth     = true;
    depthTargetDesc.debugName = "DepthCameraExample.DepthTarget";
    const cressim::neo::gpu::GpuRenderTargetHandle depthTarget =
        device->renderTargetSystem().createRenderTarget(depthTargetDesc);
    if (!device->renderTargetSystem().isValidRenderTarget(depthTarget))
    {
        runtime.shutdown();
        viewer.shutdown();
        CRESSIM_LOG_ERROR("Depth target creation failed.\n");
        return 1;
    }

    const auto depthCameraEntity = world.createEntity();
    world.setTransform(depthCameraEntity, cameraTransform);
    CameraComponent depthCamera{};
    depthCamera.product            = CameraComponent::Product::Depth;
    depthCamera.verticalFovDegrees = colorCamera.verticalFovDegrees;
    depthCamera.nearClip           = colorCamera.nearClip;
    depthCamera.farClip            = colorCamera.farClip;
    depthCamera.output.mode        = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
    depthCamera.output.binding     = cressim::neo::gpu::GpuRenderTargetBinding{depthTarget, 0u, 1u};
    depthCamera.outputWidth        = depthTargetDesc.width;
    depthCamera.outputHeight       = depthTargetDesc.height;
    depthCamera.clearColor         = false;
    depthCamera.clearDepth         = true;
    world.setCamera(depthCameraEntity, depthCamera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.55f, -1.0f, 0.25f};
    light.color     = {1.0f, 0.98f, 0.95f};
    light.intensity = 7.5f;
    world.setDirectionalLight(lightEntity, light);

    auto &resources = runtime.getResources();
    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(0.8f, "DepthCameraExample.CubeMesh"));
    const auto sphereMesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        0.7f, 32u, 16u, "DepthCameraExample.SphereMesh"));
    const auto groundMesh = resources.registerMesh(cressim::neo::examples::helpers::makePlaneMesh(
        14.0f, 14.0f, "DepthCameraExample.GroundMesh"));

    MaterialResourceDesc groundMaterial{};
    groundMaterial.debugName = "DepthCameraExample.Ground";
    groundMaterial.baseColor = {0.72f, 0.74f, 0.78f};
    groundMaterial.roughness = 0.95f;
    const auto groundMaterialHandle = resources.registerMaterial(groundMaterial);

    MaterialResourceDesc coralMaterial{};
    coralMaterial.debugName = "DepthCameraExample.Coral";
    coralMaterial.baseColor = {0.92f, 0.44f, 0.34f};
    coralMaterial.roughness = 0.55f;
    const auto coralMaterialHandle = resources.registerMaterial(coralMaterial);

    MaterialResourceDesc tealMaterial{};
    tealMaterial.debugName = "DepthCameraExample.Teal";
    tealMaterial.baseColor = {0.18f, 0.64f, 0.72f};
    tealMaterial.roughness = 0.35f;
    tealMaterial.metallic  = 0.15f;
    const auto tealMaterialHandle = resources.registerMaterial(tealMaterial);

    MaterialResourceDesc amberMaterial{};
    amberMaterial.debugName = "DepthCameraExample.Amber";
    amberMaterial.baseColor = {0.88f, 0.72f, 0.28f};
    amberMaterial.roughness = 0.42f;
    const auto amberMaterialHandle = resources.registerMaterial(amberMaterial);

    spawnRenderable(world, groundMesh, groundMaterialHandle, {0.0f, -1.0f, 6.0f},
                    {1.0f, 1.0f, 1.0f});
    spawnRenderable(world, cubeMesh, coralMaterialHandle, {-1.8f, -0.15f, 3.5f},
                    {1.0f, 1.5f, 1.0f});
    spawnRenderable(world, cubeMesh, tealMaterialHandle, {1.6f, 0.3f, 6.2f},
                    {1.25f, 1.25f, 1.25f},
                    quaternionFromEulerDegrees(18.0f, 28.0f, 0.0f));
    spawnRenderable(world, sphereMesh, amberMaterialHandle, {0.0f, 0.1f, 2.3f},
                    {1.0f, 1.0f, 1.0f});

    CRESSIM_LOG_INFO("Depth camera example ready. Camera mode shows shaded color. Press U in the "
                     "viewer to toggle the authored depth camera output.");

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [colorCameraEntity, depthCameraEntity](
                               const cressim::neo::common::FrameContext &, Runtime &runtimeRef)
    {
        auto &worldRef = runtimeRef.getWorld();
        const std::optional<TransformComponent> colorTransform =
            worldRef.tryGetTransform(colorCameraEntity);
        if (!colorTransform.has_value())
        {
            return;
        }

        worldRef.setTransform(depthCameraEntity, *colorTransform);
    };

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{colorCameraEntity}, callbacks);

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
