#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "common/logger.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuRenderTargetBinding;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;

void printUsage(const char* appName)
{
    CRESSIM_LOG_ERROR( "Usage: " , appName , " [--output path.ppm]\n");
}

bool isValidReadback(const GpuRenderTargetReadbackEvent& event)
{
    if (event.width == 0u || event.height == 0u || event.rowStrideBytes < event.width * 4u)
    {
        return false;
    }
    return event.colorBytes.size() >=
           static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height);
}

bool isNear(std::uint8_t value, std::uint8_t expected, std::uint8_t tolerance)
{
    const int diff = static_cast<int>(value) - static_cast<int>(expected);
    return diff >= -static_cast<int>(tolerance) && diff <= static_cast<int>(tolerance);
}

bool isNearColor(const std::array<std::uint8_t, 4u>& pixel, std::uint8_t r, std::uint8_t g,
                 std::uint8_t b, std::uint8_t tolerance)
{
    return isNear(pixel[0], r, tolerance) && isNear(pixel[1], g, tolerance) &&
           isNear(pixel[2], b, tolerance);
}

template <typename Predicate>
std::uint64_t countPixelsMatching(const GpuRenderTargetReadbackEvent& event, std::uint32_t xBegin,
                                  std::uint32_t xEnd, Predicate predicate)
{
    if (!isValidReadback(event) || xBegin >= xEnd || xEnd > event.width)
    {
        return 0u;
    }

    std::uint64_t count = 0u;
    for (std::uint32_t y = 0u; y < event.height; ++y)
    {
        for (std::uint32_t x = xBegin; x < xEnd; ++x)
        {
            const std::size_t offset =
                static_cast<std::size_t>(y) * event.rowStrideBytes + static_cast<std::size_t>(x) * 4u;
            const std::array<std::uint8_t, 4u> pixel{
                event.colorBytes[offset + 0u],
                event.colorBytes[offset + 1u],
                event.colorBytes[offset + 2u],
                event.colorBytes[offset + 3u]};
            if (predicate(pixel))
            {
                ++count;
            }
        }
    }
    return count;
}

bool isYellowClear(const std::array<std::uint8_t, 4u>& pixel)
{
    return isNearColor(pixel, 242u, 230u, 26u, 20u);
}

bool isBlueClear(const std::array<std::uint8_t, 4u>& pixel)
{
    return isNearColor(pixel, 26u, 51u, 242u, 20u);
}

bool writePpm(const std::string& path, const GpuRenderTargetReadbackEvent& event)
{
    if (!isValidReadback(event))
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
    {
        return false;
    }

    out << "P6\n" << event.width << " " << event.height << "\n255\n";

    std::vector<std::uint8_t> rgbRow(static_cast<std::size_t>(event.width) * 3u);
    for (std::uint32_t y = 0; y < event.height; ++y)
    {
        const std::uint8_t* src =
            event.colorBytes.data() + static_cast<std::size_t>(y) * event.rowStrideBytes;
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            rgbRow[static_cast<std::size_t>(x) * 3u + 0u] =
                src[static_cast<std::size_t>(x) * 4u + 0u];
            rgbRow[static_cast<std::size_t>(x) * 3u + 1u] =
                src[static_cast<std::size_t>(x) * 4u + 1u];
            rgbRow[static_cast<std::size_t>(x) * 3u + 2u] =
                src[static_cast<std::size_t>(x) * 4u + 2u];
        }
        out.write(reinterpret_cast<const char*>(rgbRow.data()),
                  static_cast<std::streamsize>(rgbRow.size()));
    }

    return out.good();
}

GpuRenderTargetReadbackEvent renderAndReadback(Runtime& runtime, GpuDevice& graphicsDevice,
                                               const GpuRenderTargetHandle target,
                                               FrameContext& frame)
{
    const GpuRenderTargetReadbackRequest request =
        graphicsDevice.renderTargetSystem().requestRenderTargetReadback(
            GpuRenderTargetBinding{target, 0u, 1u});
    runtime.tick(frame);

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
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;

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
    camera.output.mode         = cressim::neo::gpu::CameraOutputMode::ExplicitSurface;
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
    runtime.tick(frame);

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
    runtime.tick(frame);

    frame.frameIndex  = 2u;
    frame.timeSeconds = static_cast<double>(frame.frameIndex) * static_cast<double>(frame.deltaSeconds);
    const GpuRenderTargetReadbackEvent preservedEvent =
        renderAndReadback(runtime, *graphicsDevice, target, frame);
    if (!isValidReadback(preservedEvent))
    {
        CRESSIM_LOG_ERROR( "Expected valid readback for preserved viewport render.\n");
        runtime.shutdown();
        return 1;
    }

    const std::uint32_t midX = preservedEvent.width / 2u;
    const std::uint64_t leftYellowCount =
        countPixelsMatching(preservedEvent, 0u, midX, isYellowClear);
    const std::uint64_t leftPixelCount =
        static_cast<std::uint64_t>(midX) * preservedEvent.height;
    const std::uint64_t rightYellowCount =
        countPixelsMatching(preservedEvent, midX, preservedEvent.width, isYellowClear);
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
    if (!isValidReadback(clearedEvent))
    {
        CRESSIM_LOG_ERROR( "Expected valid readback for cleared viewport render.\n");
        runtime.shutdown();
        return 1;
    }

    const std::uint64_t clearedRightBlueCount =
        countPixelsMatching(clearedEvent, midX, clearedEvent.width, isBlueClear);
    const std::uint64_t clearedRightPixelCount =
        static_cast<std::uint64_t>(clearedEvent.width - midX) * clearedEvent.height;
    if (clearedRightBlueCount != clearedRightPixelCount)
    {
        CRESSIM_LOG_ERROR( "Expected whole-target clear semantics for viewport render when clearColor is enabled.\n");
        runtime.shutdown();
        return 1;
    }

    if (!outputPath.empty() && !writePpm(outputPath, clearedEvent))
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
