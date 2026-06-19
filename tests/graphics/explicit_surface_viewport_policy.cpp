#include "common/frame_context.h"
#include "common/logger.h"
#include "helpers/readback.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <cmath>
#include <cstdint>
#include <string>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuRenderTargetBinding;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::tests::helpers::ReadbackPixel;

void printUsage(const char* appName)
{
    CRESSIM_LOG_ERROR( "Usage: " , appName , " [--output path.ppm]\n");
}

bool isNear(float value, float expected, float tolerance)
{
    return std::fabs(value - expected) <= tolerance;
}

bool isNearColor(const ReadbackPixel& pixel, float r, float g, float b, float tolerance)
{
    return isNear(pixel.r, r, tolerance) && isNear(pixel.g, g, tolerance) &&
           isNear(pixel.b, b, tolerance);
}

bool isYellowClear(const ReadbackPixel& pixel)
{
    return isNearColor(pixel, 0.95f, 0.90f, 0.10f, 0.08f);
}

bool isBlueClear(const ReadbackPixel& pixel)
{
    return isNearColor(pixel, 0.10f, 0.20f, 0.95f, 0.08f);
}

GpuRenderTargetReadbackEvent renderAndReadback(Runtime& runtime, GpuDevice& graphicsDevice,
                                               const GpuRenderTargetHandle target,
                                               FrameContext& frame)
{
    const GpuRenderTargetReadbackRequest request =
        graphicsDevice.renderTargetSystem().requestRenderTargetReadback(
            GpuRenderTargetBinding{target, 0u, 1u});
    runtime.prepare();
    const bool physicsStepSucceeded = runtime.stepPhysics(frame);
    if (physicsStepSucceeded)
    {
        (void)runtime.stepSimulationSensors(frame);
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

    GpuRenderTargetReadbackEvent event{};
    if (request.id == 0u || !graphicsDevice.renderTargetSystem().tryGetRenderTargetReadback(request, event))
    {
        return {};
    }
    return event;
}

} // namespace

int main(int argc, char** argv)
{
    std::string outputPath;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--output")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            outputPath = argv[++i];
            continue;
        }

        printUsage(argv[0]);
        return 2;
    }

    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = cressim::neo::gpu::GpuBackend::Vulkan;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    GpuDevice* graphicsDevice = runtime.getGpuDevice();
    if (graphicsDevice == nullptr)
    {
        CRESSIM_LOG_ERROR( "Graphics device not available.\n");
        runtime.shutdown();
        return 1;
    }

    GpuRenderTargetDesc targetDesc{};
    targetDesc.width            = 128u;
    targetDesc.height           = 128u;
    targetDesc.arraySize        = 1u;
    targetDesc.layeredRendering = false;
    targetDesc.debugName        = "ViewportPolicy.Target";
    const GpuRenderTargetHandle target = graphicsDevice->renderTargetSystem().createRenderTarget(targetDesc);
    if (!graphicsDevice->renderTargetSystem().isValidRenderTarget(target))
    {
        CRESSIM_LOG_ERROR( "Failed to create explicit surface target.\n");
        runtime.shutdown();
        return 1;
    }

    auto& world = runtime.getWorld();
    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};
    CameraComponent camera{};
    camera.output.mode         = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
    camera.output.binding      = GpuRenderTargetBinding{target, 0u, 1u};
    camera.viewport            = {0.0f, 0.0f, 0.5f, 1.0f};
    camera.clearColor          = true;
    camera.clearDepth          = true;
    camera.clearColorValue     = {0.95f, 0.90f, 0.10f, 1.0f};
    camera.renderOrder         = 0u;
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {0.0f, 0.0f, -1.0f};
    world.setDirectionalLight(lightEntity, light);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex   = 0u;
    frame.timeSeconds  = 0.0;
    runtime.prepare();
    const bool secondPhysicsStepSucceeded = runtime.stepPhysics(frame);
    if (secondPhysicsStepSucceeded)
    {
        (void)runtime.stepSimulationSensors(frame);
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

    auto& resources = runtime.getResources();
    MeshResourceDesc meshDesc{};
    meshDesc.debugName = "ViewportPolicy.FullTriangle";
    meshDesc.vertices = {
        {{-4.0f, -4.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{4.0f, -4.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 4.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f}};
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "ViewportPolicy.Red";
    materialDesc.baseColor = {0.95f, 0.10f, 0.10f};
    materialDesc.metallic  = 0.0f;
    materialDesc.roughness = 0.6f;
    const auto material = resources.registerMaterial(materialDesc);

    const auto renderableEntity = world.createEntity();
    world.setTransform(renderableEntity, TransformComponent{});
    MeshRendererComponent meshRenderer{};
    meshRenderer.mesh     = mesh;
    meshRenderer.material = material;
    meshRenderer.visible  = true;
    world.setMeshRenderer(renderableEntity, meshRenderer);

    camera.clearColor = false;
    world.setCamera(cameraEntity, camera);
    frame.frameIndex  = 1u;
    frame.timeSeconds = static_cast<double>(frame.deltaSeconds);
    runtime.prepare();
    const bool thirdPhysicsStepSucceeded = runtime.stepPhysics(frame);
    if (thirdPhysicsStepSucceeded)
    {
        (void)runtime.stepSimulationSensors(frame);
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

    frame.frameIndex  = 2u;
    frame.timeSeconds = static_cast<double>(frame.frameIndex) * static_cast<double>(frame.deltaSeconds);
    const GpuRenderTargetReadbackEvent preservedEvent =
        renderAndReadback(runtime, *graphicsDevice, target, frame);
    if (!cressim::neo::tests::helpers::isValidReadback(preservedEvent))
    {
        CRESSIM_LOG_ERROR( "Expected valid readback for preserved viewport render.\n");
        runtime.shutdown();
        return 1;
    }

    const std::uint32_t midX = preservedEvent.width / 2u;
    const std::uint64_t leftYellowCount =
        cressim::neo::tests::helpers::countPixelsMatching(preservedEvent, 0u, midX, isYellowClear);
    const std::uint64_t leftPixelCount =
        static_cast<std::uint64_t>(midX) * preservedEvent.height;
    const std::uint64_t rightYellowCount =
        cressim::neo::tests::helpers::countPixelsMatching(
            preservedEvent, midX, preservedEvent.width, isYellowClear);
    const std::uint64_t rightPixelCount =
        static_cast<std::uint64_t>(preservedEvent.width - midX) * preservedEvent.height;
    if (leftYellowCount == leftPixelCount)
    {
        CRESSIM_LOG_ERROR( "Expected viewport draw to affect the left half of the explicit surface.\n");
        runtime.shutdown();
        return 1;
    }
    if (rightYellowCount != rightPixelCount)
    {
        CRESSIM_LOG_ERROR( "Expected right half of explicit surface to remain untouched when clearColor is disabled.\n");
        runtime.shutdown();
        return 1;
    }

    camera.clearColor      = true;
    camera.clearColorValue = {0.10f, 0.20f, 0.95f, 1.0f};
    world.setCamera(cameraEntity, camera);
    frame.frameIndex  = 3u;
    frame.timeSeconds = static_cast<double>(frame.frameIndex) * static_cast<double>(frame.deltaSeconds);
    const GpuRenderTargetReadbackEvent clearedEvent =
        renderAndReadback(runtime, *graphicsDevice, target, frame);
    if (!cressim::neo::tests::helpers::isValidReadback(clearedEvent))
    {
        CRESSIM_LOG_ERROR( "Expected valid readback for cleared viewport render.\n");
        runtime.shutdown();
        return 1;
    }

    const std::uint64_t clearedRightBlueCount =
        cressim::neo::tests::helpers::countPixelsMatching(
            clearedEvent, midX, clearedEvent.width, isBlueClear);
    const std::uint64_t clearedRightPixelCount =
        static_cast<std::uint64_t>(clearedEvent.width - midX) * clearedEvent.height;
    if (clearedRightBlueCount != clearedRightPixelCount)
    {
        CRESSIM_LOG_ERROR( "Expected whole-target clear semantics for viewport render when clearColor is enabled.\n");
        runtime.shutdown();
        return 1;
    }

    if (!outputPath.empty() &&
        !cressim::neo::tests::helpers::writePpm(outputPath, clearedEvent))
    {
        CRESSIM_LOG_ERROR( "Failed to write output image: " , outputPath , '\n');
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO( "Explicit surface viewport policy checks passed.\n");
    if (!outputPath.empty())
    {
        CRESSIM_LOG_INFO( "Wrote viewport policy image to " , outputPath , '\n');
    }
    return 0;
}
