#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
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
using cressim::neo::graphics::GraphicsBackend;
using cressim::neo::graphics::GraphicsDevice;
using cressim::neo::graphics::RenderTargetDesc;
using cressim::neo::graphics::RenderTargetHandle;
using cressim::neo::graphics::RenderTargetReadbackEvent;
using cressim::neo::graphics::RenderTargetReadbackRequest;

GraphicsBackend parseBackend(const std::string& value)
{
    if (value == "null")
    {
        return GraphicsBackend::Null;
    }
    if (value == "vulkan")
    {
        return GraphicsBackend::Vulkan;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

void printUsage(const char* appName)
{
    std::cerr << "Usage: " << appName << " [--backend vulkan|null] [--frames N] [--validation on|off]\n";
}

bool isNear(std::uint8_t value, std::uint8_t expected, std::uint8_t tolerance)
{
    const int diff = static_cast<int>(value) - static_cast<int>(expected);
    return diff >= -static_cast<int>(tolerance) && diff <= static_cast<int>(tolerance);
}

bool containsNonClearPixel(const RenderTargetReadbackEvent& event)
{
    if (event.width == 0 || event.height == 0 || event.rowStrideBytes < event.width * 4u)
    {
        return false;
    }
    if (event.colorBytes.size() < static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height))
    {
        return false;
    }

    constexpr std::uint8_t kClearR = 5;
    constexpr std::uint8_t kClearG = 5;
    constexpr std::uint8_t kClearB = 8;
    constexpr std::uint8_t kClearA = 255;
    constexpr std::uint8_t kTolerance = 2;

    for (std::uint32_t y = 0; y < event.height; ++y)
    {
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            const std::size_t offset = static_cast<std::size_t>(y) * event.rowStrideBytes + static_cast<std::size_t>(x) * 4u;
            const std::uint8_t r = event.colorBytes[offset + 0u];
            const std::uint8_t g = event.colorBytes[offset + 1u];
            const std::uint8_t b = event.colorBytes[offset + 2u];
            const std::uint8_t a = event.colorBytes[offset + 3u];

            const bool nearClear =
                isNear(r, kClearR, kTolerance) &&
                isNear(g, kClearG, kTolerance) &&
                isNear(b, kClearB, kTolerance) &&
                isNear(a, kClearA, kTolerance);

            if (!nearClear)
            {
                return true;
            }
        }
    }

    return false;
}

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.graphicsDeviceDesc.preferredBackend = GraphicsBackend::Vulkan;
    config.graphicsDeviceDesc.enableValidation = false;

    std::uint64_t numFrames = 3;

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
            config.graphicsDeviceDesc.preferredBackend = parseBackend(argv[++i]);
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
        if (arg == "--validation")
        {
            if (i + 1 >= argc)
            {
                printUsage(argv[0]);
                return 2;
            }
            const std::string value = argv[++i];
            if (value == "on")
            {
                config.graphicsDeviceDesc.enableValidation = true;
                continue;
            }
            if (value == "off")
            {
                config.graphicsDeviceDesc.enableValidation = false;
                continue;
            }
            printUsage(argv[0]);
            return 2;
        }

        printUsage(argv[0]);
        return 2;
    }

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        std::cerr << "Runtime initialization failed.\n";
        return 1;
    }

    auto& world = runtime.getWorld();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};
    CameraComponent camera{};
    camera.renderOrder = 0;
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, camera);

    GraphicsDevice* graphicsDevice = runtime.getGraphicsDevice();
    RenderTargetHandle secondaryTarget{};
    if (graphicsDevice != nullptr)
    {
        RenderTargetDesc secondaryTargetDesc{};
        secondaryTargetDesc.width = 640;
        secondaryTargetDesc.height = 480;
        secondaryTargetDesc.debugName = "Smoke.SecondaryCamera";
        // This enables readback event plumbing in the smoke test.
        secondaryTargetDesc.cpuReadback = true;
        secondaryTarget = graphicsDevice->createRenderTarget(secondaryTargetDesc);
    }

    if (graphicsDevice != nullptr && graphicsDevice->isValidRenderTarget(secondaryTarget))
    {
        const auto secondaryCameraEntity = world.createEntity();
        TransformComponent secondaryCameraTransform{};
        secondaryCameraTransform.worldTransform.position = {-1.0f, 1.5f, -2.5f};

        CameraComponent secondaryCamera{};
        secondaryCamera.outputTarget = secondaryTarget;
        secondaryCamera.outputWidth = 800;
        secondaryCamera.outputHeight = 600;
        secondaryCamera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
        secondaryCamera.renderOrder = 1;

        world.setTransform(secondaryCameraEntity, secondaryCameraTransform);
        world.setCamera(secondaryCameraEntity, secondaryCamera);
    }

    const auto lightEntity = world.createEntity();
    world.setDirectionalLight(lightEntity, DirectionalLightComponent{});

    auto& resources = runtime.getResources();
    cressim::neo::graphics::MeshResourceDesc meshDesc{};
    meshDesc.debugName = "Smoke.TriangleMesh";
    meshDesc.vertices = {
        {{-0.6f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.6f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 0.6f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f}};
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    cressim::neo::graphics::MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "Smoke.TriangleMaterial";
    materialDesc.baseColor = {0.95f, 0.25f, 0.20f};
    materialDesc.metallic = 0.1f;
    materialDesc.roughness = 0.35f;
    const auto material = resources.registerMaterial(materialDesc);

    const auto renderableEntity = world.createEntity();
    world.setTransform(renderableEntity, TransformComponent{});
    MeshRendererComponent meshRenderer{};
    meshRenderer.mesh = mesh;
    meshRenderer.material = material;
    meshRenderer.visible = true;
    world.setMeshRenderer(renderableEntity, meshRenderer);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    std::vector<RenderTargetReadbackRequest> readbackRequests;

    for (std::uint64_t i = 0; i < numFrames; ++i)
    {
        if (graphicsDevice != nullptr && graphicsDevice->isValidRenderTarget(secondaryTarget))
        {
            const RenderTargetReadbackRequest request = graphicsDevice->requestRenderTargetReadback(secondaryTarget);
            if (request.id != 0)
            {
                readbackRequests.push_back(request);
            }
        }
        frame.frameIndex = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.tick(frame);
    }

    std::uint32_t readbackEvents = 0;
    std::uint32_t payloadEvents = 0;
    bool foundNonClearPixel = false;
    if (graphicsDevice != nullptr)
    {
        for (const RenderTargetReadbackRequest request : readbackRequests)
        {
            RenderTargetReadbackEvent event{};
            if (!graphicsDevice->tryGetRenderTargetReadback(request, event))
            {
                continue;
            }
            ++readbackEvents;
            if (!event.colorBytes.empty())
            {
                ++payloadEvents;
                foundNonClearPixel = foundNonClearPixel || containsNonClearPixel(event);
            }
        }
    }

    if (config.graphicsDeviceDesc.preferredBackend == GraphicsBackend::Vulkan)
    {
        if (readbackEvents == 0)
        {
            runtime.shutdown();
            std::cerr << "Smoke run failed: expected at least one readback event.\n";
            return 1;
        }
        if (payloadEvents == 0)
        {
            runtime.shutdown();
            std::cerr << "Smoke run failed: expected readback payload for Vulkan backend.\n";
            return 1;
        }
        if (!foundNonClearPixel)
        {
            runtime.shutdown();
            std::cerr << "Smoke run failed: readback payload contains only clear color.\n";
            return 1;
        }
    }

    runtime.shutdown();

    std::cout << "Smoke run passed. Frames: " << numFrames << ", Readback events: " << readbackEvents
              << ", Payload events: " << payloadEvents << '\n';
    return 0;
}
