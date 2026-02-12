#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::GraphicsBackend;
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
    std::cerr << "Usage: " << appName
              << " [--backend vulkan|null] [--frames N] [--output path.ppm] [--validation on|off]\n";
}

bool isNear(std::uint8_t value, std::uint8_t expected, std::uint8_t tolerance)
{
    const int diff = static_cast<int>(value) - static_cast<int>(expected);
    return diff >= -static_cast<int>(tolerance) && diff <= static_cast<int>(tolerance);
}

bool containsNonClearPixel(const RenderTargetReadbackEvent& event)
{
    constexpr std::uint8_t kClearR = 5;
    constexpr std::uint8_t kClearG = 5;
    constexpr std::uint8_t kClearB = 8;
    constexpr std::uint8_t kClearA = 255;
    constexpr std::uint8_t kTolerance = 2;

    if (event.width == 0 || event.height == 0 || event.rowStrideBytes < event.width * 4u)
    {
        return false;
    }
    if (event.colorRgba8.size() < static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height))
    {
        return false;
    }

    for (std::uint32_t y = 0; y < event.height; ++y)
    {
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            const std::size_t offset = static_cast<std::size_t>(y) * event.rowStrideBytes + static_cast<std::size_t>(x) * 4u;
            const std::uint8_t r = event.colorRgba8[offset + 0u];
            const std::uint8_t g = event.colorRgba8[offset + 1u];
            const std::uint8_t b = event.colorRgba8[offset + 2u];
            const std::uint8_t a = event.colorRgba8[offset + 3u];

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

bool writePpm(const std::string& path, const RenderTargetReadbackEvent& event)
{
    if (event.width == 0 || event.height == 0 || event.rowStrideBytes < event.width * 4u)
    {
        return false;
    }
    if (event.colorRgba8.size() < static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height))
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
        const std::uint8_t* src = event.colorRgba8.data() + static_cast<std::size_t>(y) * event.rowStrideBytes;
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            rgbRow[static_cast<std::size_t>(x) * 3u + 0u] = src[static_cast<std::size_t>(x) * 4u + 0u];
            rgbRow[static_cast<std::size_t>(x) * 3u + 1u] = src[static_cast<std::size_t>(x) * 4u + 1u];
            rgbRow[static_cast<std::size_t>(x) * 3u + 2u] = src[static_cast<std::size_t>(x) * 4u + 2u];
        }
        out.write(reinterpret_cast<const char*>(rgbRow.data()), static_cast<std::streamsize>(rgbRow.size()));
    }

    return out.good();
}

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.graphics.preferredBackend = GraphicsBackend::Vulkan;
    config.graphics.enableValidation = false;

    std::uint64_t numFrames = 2;
    std::string outputPath = "triangle_readback.ppm";

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
    auto* graphicsDevice = runtime.getGraphicsDevice();
    if (graphicsDevice == nullptr)
    {
        runtime.shutdown();
        std::cerr << "Graphics device unavailable.\n";
        return 1;
    }

    RenderTargetDesc targetDesc{};
    targetDesc.width = 512;
    targetDesc.height = 512;
    targetDesc.cpuReadback = true;
    targetDesc.debugName = "TriangleCapture.Target";
    const RenderTargetHandle target = graphicsDevice->createRenderTarget(targetDesc);
    if (!graphicsDevice->isValidRenderTarget(target))
    {
        runtime.shutdown();
        std::cerr << "Failed to create readback target.\n";
        return 1;
    }

    const auto cameraEntity = world.createEntity();
    world.setTransform(cameraEntity, TransformComponent{});
    CameraComponent camera{};
    camera.outputTarget = target;
    camera.outputWidth = targetDesc.width;
    camera.outputHeight = targetDesc.height;
    camera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
    camera.requestReadback = true;
    world.setCamera(cameraEntity, camera);

    auto& resources = runtime.getScene().resources();
    cressim::neo::graphics::MeshResourceDesc meshDesc{};
    meshDesc.debugName = "TriangleCapture.Mesh";
    cressim::neo::graphics::MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "TriangleCapture.Material";

    const auto renderableEntity = world.createEntity();
    MeshRendererComponent meshRenderer{};
    meshRenderer.mesh = resources.registerMesh(meshDesc);
    meshRenderer.material = resources.registerMaterial(materialDesc);
    meshRenderer.visible = true;
    world.setTransform(renderableEntity, TransformComponent{});
    world.setMeshRenderer(renderableEntity, meshRenderer);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    for (std::uint64_t i = 0; i < numFrames; ++i)
    {
        frame.frameIndex = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.tick(frame);
    }

    RenderTargetReadbackEvent captured{};
    bool hasCapturedPayload = false;
    RenderTargetReadbackEvent event{};
    while (runtime.tryPopReadbackEvent(event))
    {
        if (event.target.id == target.id && !event.colorRgba8.empty())
        {
            captured = event;
            hasCapturedPayload = true;
        }
    }

    runtime.shutdown();

    if (config.graphics.preferredBackend == GraphicsBackend::Null)
    {
        std::cout << "Null backend selected; capture skipped.\n";
        return 0;
    }

    if (!hasCapturedPayload)
    {
        std::cerr << "No readback payload captured.\n";
        return 1;
    }
    if (!containsNonClearPixel(captured))
    {
        std::cerr << "Captured image appears to contain only clear color.\n";
        return 1;
    }
    if (!writePpm(outputPath, captured))
    {
        std::cerr << "Failed to write image: " << outputPath << '\n';
        return 1;
    }

    std::cout << "Triangle capture passed. Wrote " << captured.width << "x" << captured.height
              << " image to " << outputPath << '\n';
    return 0;
}
