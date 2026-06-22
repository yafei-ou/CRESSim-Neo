#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/example_cli.h"
#include "helpers/shape_meshes.h"
#include "helpers/viewer_example.h"
#include "viewer/debug_viewer_app.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

enum class SensorProductMode
{
    ColorDepth,
    DepthOnly,
    SegmentationDepth,
    All,
};

struct ExampleOptions
{
    CommonExampleOptions common{};
    SensorProductMode sensorProducts = SensorProductMode::All;
};

void printUsage(const char *appName)
{
    cressim::neo::examples::helpers::printUsage(
        appName, " [--sensor-product colordepth|depth|segmentation|all]", true);
}

SensorProductMode parseSensorProductMode(const std::string &value)
{
    if (value == "colordepth" || value == "color-depth")
    {
        return SensorProductMode::ColorDepth;
    }
    if (value == "depth" || value == "depth-only")
    {
        return SensorProductMode::DepthOnly;
    }
    if (value == "segmentation" || value == "segmentation-depth")
    {
        return SensorProductMode::SegmentationDepth;
    }
    if (value == "all" || value == "both")
    {
        return SensorProductMode::All;
    }

    throw std::invalid_argument("Unsupported --sensor-product value: " + value);
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
                     std::uint32_t envIndex,
                     cressim::neo::graphics::MaterialHandle material,
                     std::uint32_t segmentationId,
                     const Diligent::float3 &position, const Diligent::float3 &scale,
                     const Diligent::QuaternionF &rotation = Diligent::QuaternionF{})
{
    const auto entity = world.createEntity(envIndex);
    TransformComponent transform{};
    transform.worldTransform.position = position;
    transform.worldTransform.scale    = scale;
    transform.worldTransform.rotation = rotation;
    world.setTransform(entity, transform);

    MeshRendererComponent renderer{};
    renderer.mesh           = mesh;
    renderer.material       = material;
    renderer.segmentationId = segmentationId;
    world.setMeshRenderer(entity, renderer);
}

bool wantsColorDepthSensor(SensorProductMode mode)
{
    return mode == SensorProductMode::ColorDepth || mode == SensorProductMode::All;
}

bool wantsDepthSensor(SensorProductMode mode)
{
    return mode == SensorProductMode::DepthOnly || mode == SensorProductMode::All;
}

bool wantsSegmentationSensor(SensorProductMode mode)
{
    return mode == SensorProductMode::SegmentationDepth || mode == SensorProductMode::All;
}

void syncSensorCamera(Runtime &runtime, cressim::neo::common::EntityId sourceCameraEntity,
                      cressim::neo::common::EntityId sensorCameraEntity)
{
    auto &world = runtime.getWorld();
    const std::optional<TransformComponent> sourceTransform =
        world.tryGetTransform(sourceCameraEntity);
    const std::optional<CameraComponent> sourceCamera = world.tryGetCamera(sourceCameraEntity);
    const std::optional<CameraComponent> sensorCamera = world.tryGetCamera(sensorCameraEntity);
    if (!sourceTransform.has_value() || !sourceCamera.has_value() || !sensorCamera.has_value())
    {
        return;
    }

    world.setTransform(sensorCameraEntity, *sourceTransform);

    CameraComponent updated = *sensorCamera;
    updated.verticalFovDegrees = sourceCamera->verticalFovDegrees;
    updated.nearClip           = sourceCamera->nearClip;
    updated.farClip            = sourceCamera->farClip;
    world.setCamera(sensorCameraEntity, updated);
}

} // namespace

int main(int argc, char **argv)
{
    ExampleOptions options{};

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (cressim::neo::examples::helpers::tryParseCommonArgument(
                    argc, argv, i, options.common, true))
            {
                continue;
            }

            const std::string arg = argv[i];
            if (arg == "--sensor-product")
            {
                options.sensorProducts =
                    parseSensorProductMode(cressim::neo::examples::helpers::requireOptionValue(
                        argc, argv, i, "--sensor-product"));
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

    auto config = cressim::neo::examples::helpers::makeRuntimeConfig(options.common);
    const std::uint32_t environmentCount = std::max(options.common.envCount, 3u);
    config.sceneLayout.envCount          = environmentCount;

    DebugViewerApp viewer;
    ViewerExampleDefaults viewerDefaults{};
    viewerDefaults.windowTitle = "CRESSim Neo Camera Outputs Example";
    viewerDefaults.showStats   = true;
    const auto viewerDesc =
        cressim::neo::examples::helpers::makeViewerDesc(options.common, viewerDefaults);

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

    cressim::neo::gpu::GpuRenderTargetHandle colorDepthTarget{};
    if (wantsColorDepthSensor(options.sensorProducts))
    {
        cressim::neo::gpu::GpuRenderTargetDesc colorDepthTargetDesc{};
        colorDepthTargetDesc.width     = 960u;
        colorDepthTargetDesc.height    = 540u;
        colorDepthTargetDesc.arraySize = environmentCount;
        colorDepthTargetDesc.color     = true;
        colorDepthTargetDesc.depth     = true;
        colorDepthTargetDesc.debugName = "CameraOutputsExample.ColorDepthTarget";
        colorDepthTarget = device->renderTargetSystem().createRenderTarget(colorDepthTargetDesc);
        if (!device->renderTargetSystem().isValidRenderTarget(colorDepthTarget))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("ColorDepth target creation failed.\n");
            return 1;
        }
    }

    cressim::neo::gpu::GpuRenderTargetHandle depthTarget{};
    if (wantsDepthSensor(options.sensorProducts))
    {
        cressim::neo::gpu::GpuRenderTargetDesc depthTargetDesc{};
        depthTargetDesc.width     = 960u;
        depthTargetDesc.height    = 540u;
        depthTargetDesc.arraySize = environmentCount;
        depthTargetDesc.color     = false;
        depthTargetDesc.depth     = true;
        depthTargetDesc.debugName = "CameraOutputsExample.DepthTarget";
        depthTarget = device->renderTargetSystem().createRenderTarget(depthTargetDesc);
        if (!device->renderTargetSystem().isValidRenderTarget(depthTarget))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("Depth target creation failed.\n");
            return 1;
        }
    }

    cressim::neo::gpu::GpuRenderTargetHandle segmentationTarget{};
    if (wantsSegmentationSensor(options.sensorProducts))
    {
        cressim::neo::gpu::GpuRenderTargetDesc segmentationTargetDesc{};
        segmentationTargetDesc.width       = 960u;
        segmentationTargetDesc.height      = 540u;
        segmentationTargetDesc.arraySize   = environmentCount;
        segmentationTargetDesc.color       = true;
        segmentationTargetDesc.depth       = true;
        segmentationTargetDesc.colorFormat = Diligent::TEX_FORMAT_R32_UINT;
        segmentationTargetDesc.debugName   = "CameraOutputsExample.SegmentationDepthTarget";
        segmentationTarget =
            device->renderTargetSystem().createRenderTarget(segmentationTargetDesc);
        if (!device->renderTargetSystem().isValidRenderTarget(segmentationTarget))
        {
            runtime.shutdown();
            viewer.shutdown();
            CRESSIM_LOG_ERROR("SegmentationDepth target creation failed.\n");
            return 1;
        }
    }

    auto &resources = runtime.getResources();
    const auto cubeMesh = resources.registerMesh(
        cressim::neo::examples::helpers::makeCubeMesh(0.8f, "CameraOutputsExample.CubeMesh"));
    const auto sphereMesh = resources.registerMesh(cressim::neo::examples::helpers::makeSphereMesh(
        0.7f, 32u, 16u, "CameraOutputsExample.SphereMesh"));
    const auto groundMesh = resources.registerMesh(cressim::neo::examples::helpers::makePlaneMesh(
        14.0f, 14.0f, "CameraOutputsExample.GroundMesh"));

    MaterialResourceDesc groundMaterial{};
    groundMaterial.debugName = "CameraOutputsExample.Ground";
    groundMaterial.baseColor = {0.72f, 0.74f, 0.78f};
    groundMaterial.roughness = 0.95f;
    const auto groundMaterialHandle = resources.registerMaterial(groundMaterial);

    MaterialResourceDesc coralMaterial{};
    coralMaterial.debugName = "CameraOutputsExample.Coral";
    coralMaterial.baseColor = {0.92f, 0.44f, 0.34f};
    coralMaterial.roughness = 0.55f;
    const auto coralMaterialHandle = resources.registerMaterial(coralMaterial);

    MaterialResourceDesc tealMaterial{};
    tealMaterial.debugName = "CameraOutputsExample.Teal";
    tealMaterial.baseColor = {0.18f, 0.64f, 0.72f};
    tealMaterial.roughness = 0.35f;
    tealMaterial.metallic  = 0.15f;
    const auto tealMaterialHandle = resources.registerMaterial(tealMaterial);

    MaterialResourceDesc amberMaterial{};
    amberMaterial.debugName = "CameraOutputsExample.Amber";
    amberMaterial.baseColor = {0.88f, 0.72f, 0.28f};
    amberMaterial.roughness = 0.42f;
    const auto amberMaterialHandle = resources.registerMaterial(amberMaterial);

    std::vector<cressim::neo::common::EntityId> viewerCameraEntities;
    std::vector<cressim::neo::common::EntityId> colorDepthSensorEntities;
    std::vector<cressim::neo::common::EntityId> depthSensorEntities;
    std::vector<cressim::neo::common::EntityId> segmentationSensorEntities;
    viewerCameraEntities.reserve(environmentCount);
    colorDepthSensorEntities.reserve(environmentCount);
    depthSensorEntities.reserve(environmentCount);
    segmentationSensorEntities.reserve(environmentCount);

    for (std::uint32_t envIndex = 0u; envIndex < environmentCount; ++envIndex)
    {
        const float envOffset = (static_cast<float>(envIndex) -
                                 0.5f * static_cast<float>(environmentCount - 1u)) *
                                18.0f;
        const float cameraYaw    = -8.0f + static_cast<float>(envIndex) * 6.0f;
        const float cameraHeight = 2.0f + static_cast<float>(envIndex) * 0.18f;
        const float farBias      = static_cast<float>(envIndex) * 3.0f;

        const auto viewerCameraEntity = world.createEntity(envIndex);
        TransformComponent cameraTransform{};
        cameraTransform.worldTransform.position = {envOffset, cameraHeight, -8.0f};
        cameraTransform.worldTransform.rotation =
            quaternionFromEulerDegrees(-4.0f, cameraYaw, 0.0f);
        world.setTransform(viewerCameraEntity, cameraTransform);

        CameraComponent viewerCamera{};
        viewerCamera.verticalFovDegrees = 50.0f + static_cast<float>(envIndex) * 2.0f;
        viewerCamera.nearClip           = 0.05f;
        viewerCamera.farClip            = 40.0f + farBias;
        world.setCamera(viewerCameraEntity, viewerCamera);
        viewerCameraEntities.push_back(viewerCameraEntity);

        if (wantsColorDepthSensor(options.sensorProducts))
        {
            const auto sensorEntity = world.createEntity(envIndex);
            world.setTransform(sensorEntity, cameraTransform);

            CameraComponent sensorCamera{};
            sensorCamera.product            = CameraComponent::Product::ColorDepth;
            sensorCamera.verticalFovDegrees = viewerCamera.verticalFovDegrees;
            sensorCamera.nearClip           = viewerCamera.nearClip;
            sensorCamera.farClip            = viewerCamera.farClip;
            sensorCamera.output.mode        = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
            sensorCamera.output.binding =
                cressim::neo::gpu::GpuRenderTargetBinding{colorDepthTarget, envIndex, 1u};
            sensorCamera.outputWidth  = 960u;
            sensorCamera.outputHeight = 540u;
            sensorCamera.clearColor   = true;
            sensorCamera.clearDepth   = true;
            world.setCamera(sensorEntity, sensorCamera);
            colorDepthSensorEntities.push_back(sensorEntity);
        }

        if (wantsDepthSensor(options.sensorProducts))
        {
            const auto sensorEntity = world.createEntity(envIndex);
            world.setTransform(sensorEntity, cameraTransform);

            CameraComponent sensorCamera{};
            sensorCamera.product            = CameraComponent::Product::Depth;
            sensorCamera.verticalFovDegrees = viewerCamera.verticalFovDegrees;
            sensorCamera.nearClip           = viewerCamera.nearClip;
            sensorCamera.farClip            = viewerCamera.farClip;
            sensorCamera.output.mode        = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
            sensorCamera.output.binding =
                cressim::neo::gpu::GpuRenderTargetBinding{depthTarget, envIndex, 1u};
            sensorCamera.outputWidth  = 960u;
            sensorCamera.outputHeight = 540u;
            sensorCamera.clearColor   = false;
            sensorCamera.clearDepth   = true;
            world.setCamera(sensorEntity, sensorCamera);
            depthSensorEntities.push_back(sensorEntity);
        }

        if (wantsSegmentationSensor(options.sensorProducts))
        {
            const auto sensorEntity = world.createEntity(envIndex);
            world.setTransform(sensorEntity, cameraTransform);

            CameraComponent sensorCamera{};
            sensorCamera.product            = CameraComponent::Product::SegmentationDepth;
            sensorCamera.verticalFovDegrees = viewerCamera.verticalFovDegrees;
            sensorCamera.nearClip           = viewerCamera.nearClip;
            sensorCamera.farClip            = viewerCamera.farClip;
            sensorCamera.output.mode        = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
            sensorCamera.output.binding =
                cressim::neo::gpu::GpuRenderTargetBinding{segmentationTarget, envIndex, 1u};
            sensorCamera.outputWidth  = 960u;
            sensorCamera.outputHeight = 540u;
            sensorCamera.clearColor   = true;
            sensorCamera.clearDepth   = true;
            world.setCamera(sensorEntity, sensorCamera);
            segmentationSensorEntities.push_back(sensorEntity);
        }

        const auto lightEntity = world.createEntity(envIndex);
        DirectionalLightComponent light{};
        light.direction = {-0.55f + 0.08f * static_cast<float>(envIndex), -1.0f,
                           0.25f - 0.04f * static_cast<float>(envIndex)};
        light.color = {1.0f, 0.98f - 0.03f * static_cast<float>(envIndex),
                       0.95f - 0.04f * static_cast<float>(envIndex)};
        light.intensity = 7.5f + 0.6f * static_cast<float>(envIndex);
        world.setDirectionalLight(lightEntity, light);

        spawnRenderable(world, groundMesh, envIndex, groundMaterialHandle, 1u,
                        {envOffset, -1.0f, 6.0f}, {1.0f, 1.0f, 1.0f});
        spawnRenderable(world, cubeMesh, envIndex, coralMaterialHandle, 2u,
                        {envOffset - 1.8f, -0.15f, 3.5f + 0.35f * static_cast<float>(envIndex)},
                        {1.0f, 1.5f, 1.0f},
                        quaternionFromEulerDegrees(0.0f, 8.0f * static_cast<float>(envIndex), 0.0f));
        spawnRenderable(world, cubeMesh, envIndex, tealMaterialHandle, 3u,
                        {envOffset + 1.6f, 0.3f, 6.2f - 0.45f * static_cast<float>(envIndex)},
                        {1.1f + 0.12f * static_cast<float>(envIndex),
                         1.1f + 0.12f * static_cast<float>(envIndex),
                         1.1f + 0.12f * static_cast<float>(envIndex)},
                        quaternionFromEulerDegrees(18.0f + 4.0f * static_cast<float>(envIndex),
                                                   28.0f - 6.0f * static_cast<float>(envIndex), 0.0f));
        spawnRenderable(world, sphereMesh, envIndex, amberMaterialHandle, 4u,
                        {envOffset, 0.1f + 0.08f * static_cast<float>(envIndex),
                         2.3f + 0.25f * static_cast<float>(envIndex)},
                        {1.0f, 1.0f, 1.0f});
    }

    const char *modeLabel = options.sensorProducts == SensorProductMode::ColorDepth
                                ? "ColorDepth"
                                : (options.sensorProducts == SensorProductMode::DepthOnly
                                       ? "Depth"
                                       : (options.sensorProducts ==
                                                  SensorProductMode::SegmentationDepth
                                              ? "SegmentationDepth"
                                              : "All"));
    CRESSIM_LOG_INFO("Camera outputs example ready with ", environmentCount,
                     " environments and sensor mode=", modeLabel,
                     ". Camera mode shows the managed-primary viewer camera. "
                     "Press U to switch to explicit sensor outputs, then use , and . to cycle "
                     "between ColorDepth color, ColorDepth depth, Depth-only, "
                     "SegmentationDepth segmentation, and SegmentationDepth depth outputs when present.");

    DebugViewerCallbacks callbacks{};
    callbacks.beforeTick = [viewerCameraEntities, colorDepthSensorEntities, depthSensorEntities,
                            segmentationSensorEntities](
                               const cressim::neo::common::FrameContext &, Runtime &runtimeRef)
    {
        const std::size_t colorDepthCount =
            std::min(viewerCameraEntities.size(), colorDepthSensorEntities.size());
        for (std::size_t index = 0u; index < colorDepthCount; ++index)
        {
            syncSensorCamera(runtimeRef, viewerCameraEntities[index], colorDepthSensorEntities[index]);
        }

        const std::size_t depthCount =
            std::min(viewerCameraEntities.size(), depthSensorEntities.size());
        for (std::size_t index = 0u; index < depthCount; ++index)
        {
            syncSensorCamera(runtimeRef, viewerCameraEntities[index], depthSensorEntities[index]);
        }

        const std::size_t segmentationCount =
            std::min(viewerCameraEntities.size(), segmentationSensorEntities.size());
        for (std::size_t index = 0u; index < segmentationCount; ++index)
        {
            syncSensorCamera(runtimeRef, viewerCameraEntities[index],
                             segmentationSensorEntities[index]);
        }
    };

    const bool ran = viewer.run(runtime, DebugViewerCameraBinding{viewerCameraEntities.front()},
                                callbacks);

    runtime.shutdown();
    viewer.shutdown();
    return ran ? 0 : 1;
}
