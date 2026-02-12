#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::GraphicsBackend;
using cressim::neo::graphics::IGraphicsDevice;
using cressim::neo::graphics::RenderTargetDesc;
using cressim::neo::graphics::RenderTargetHandle;
using cressim::neo::graphics::RenderTargetReadbackEvent;

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

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.graphics.preferredBackend = GraphicsBackend::Vulkan;
    config.graphics.enableValidation = false;

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
            config.graphics.preferredBackend = parseBackend(argv[++i]);
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
                config.graphics.enableValidation = true;
                continue;
            }
            if (value == "off")
            {
                config.graphics.enableValidation = false;
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
    cameraTransform.worldTransform.position = {0.0f, 1.0f, 3.0f};
    CameraComponent camera{};
    camera.renderOrder = 0;
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, camera);

    IGraphicsDevice* graphicsDevice = runtime.getGraphicsDevice();
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
        secondaryCameraTransform.worldTransform.position = {-1.0f, 1.5f, 2.5f};

        CameraComponent secondaryCamera{};
        secondaryCamera.outputTarget = secondaryTarget;
        secondaryCamera.outputWidth = 800;
        secondaryCamera.outputHeight = 600;
        secondaryCamera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
        secondaryCamera.renderOrder = 1;
        secondaryCamera.requestReadback = true;

        world.setTransform(secondaryCameraEntity, secondaryCameraTransform);
        world.setCamera(secondaryCameraEntity, secondaryCamera);
    }

    const auto lightEntity = world.createEntity();
    world.setDirectionalLight(lightEntity, DirectionalLightComponent{});

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    for (std::uint64_t i = 0; i < numFrames; ++i)
    {
        frame.frameIndex = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.tick(frame);
    }

    std::uint32_t readbackEvents = 0;
    if (graphicsDevice != nullptr)
    {
        RenderTargetReadbackEvent event{};
        // Current events represent logical completion metadata, not CPU pixel payload.
        while (runtime.tryPopReadbackEvent(event))
        {
            ++readbackEvents;
        }
    }

    runtime.shutdown();

    std::cout << "Smoke run passed. Frames: " << numFrames << ", Readback events: " << readbackEvents << '\n';
    return 0;
}
