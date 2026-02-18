#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::common::Quatf;
using cressim::neo::common::Vec3f;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::GraphicsBackend;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
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
    std::cerr << "Usage: " << appName
              << " [--backend vulkan|null] [--frames N] [--output path.ppm] [--validation on|off]\n";
}

bool isNear(std::uint8_t value, std::uint8_t expected, std::uint8_t tolerance)
{
    const int diff = static_cast<int>(value) - static_cast<int>(expected);
    return diff >= -static_cast<int>(tolerance) && diff <= static_cast<int>(tolerance);
}

float degreesToRadians(float value)
{
    return value * 0.017453292519943295769f;
}

Quatf quaternionFromEulerDegrees(float pitchDegrees, float yawDegrees, float rollDegrees)
{
    const float pitch = degreesToRadians(pitchDegrees) * 0.5f;
    const float yaw = degreesToRadians(yawDegrees) * 0.5f;
    const float roll = degreesToRadians(rollDegrees) * 0.5f;

    const float sinPitch = std::sin(pitch);
    const float cosPitch = std::cos(pitch);
    const float sinYaw = std::sin(yaw);
    const float cosYaw = std::cos(yaw);
    const float sinRoll = std::sin(roll);
    const float cosRoll = std::cos(roll);

    Quatf q{};
    q.w = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw;
    q.x = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw;
    q.y = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw;
    q.z = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw;
    return q;
}

bool isValidReadback(const RenderTargetReadbackEvent& event)
{
    if (event.width == 0 || event.height == 0 || event.rowStrideBytes < event.width * 4u)
    {
        return false;
    }
    if (event.colorBytes.size() < static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height))
    {
        return false;
    }
    return true;
}

bool containsNonClearPixel(const RenderTargetReadbackEvent& event)
{
    constexpr std::uint8_t kClearR = 5;
    constexpr std::uint8_t kClearG = 5;
    constexpr std::uint8_t kClearB = 8;
    constexpr std::uint8_t kClearA = 255;
    constexpr std::uint8_t kTolerance = 2;

    if (!isValidReadback(event))
    {
        return false;
    }

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

std::array<std::uint8_t, 4> readCenterPixel(const RenderTargetReadbackEvent& event)
{
    std::array<std::uint8_t, 4> pixel{0u, 0u, 0u, 0u};
    if (!isValidReadback(event))
    {
        return pixel;
    }

    const std::uint32_t x = event.width / 2u;
    const std::uint32_t y = event.height / 2u;
    const std::size_t offset = static_cast<std::size_t>(y) * event.rowStrideBytes + static_cast<std::size_t>(x) * 4u;

    pixel[0] = event.colorBytes[offset + 0u];
    pixel[1] = event.colorBytes[offset + 1u];
    pixel[2] = event.colorBytes[offset + 2u];
    pixel[3] = event.colorBytes[offset + 3u];
    return pixel;
}

bool isRedDominant(const std::array<std::uint8_t, 4>& pixel)
{
    constexpr int kDominance = 10;
    const int r = static_cast<int>(pixel[0]);
    const int g = static_cast<int>(pixel[1]);
    const int b = static_cast<int>(pixel[2]);
    return r > g + kDominance && r > b + kDominance;
}

bool isGreenDominant(const std::array<std::uint8_t, 4>& pixel)
{
    constexpr int kDominance = 10;
    const int r = static_cast<int>(pixel[0]);
    const int g = static_cast<int>(pixel[1]);
    const int b = static_cast<int>(pixel[2]);
    return g > r + kDominance && g > b + kDominance;
}

struct DominantPixelStats
{
    std::uint64_t nonClearCount = 0;
    std::uint64_t redDominantCount = 0;
    std::uint64_t greenDominantCount = 0;
};

DominantPixelStats analyzeDominantPixels(const RenderTargetReadbackEvent& event)
{
    DominantPixelStats stats{};
    if (!isValidReadback(event))
    {
        return stats;
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
            const std::array<std::uint8_t, 4> pixel{
                event.colorBytes[offset + 0u],
                event.colorBytes[offset + 1u],
                event.colorBytes[offset + 2u],
                event.colorBytes[offset + 3u]};

            const bool nearClear =
                isNear(pixel[0], kClearR, kTolerance) &&
                isNear(pixel[1], kClearG, kTolerance) &&
                isNear(pixel[2], kClearB, kTolerance) &&
                isNear(pixel[3], kClearA, kTolerance);

            if (nearClear)
            {
                continue;
            }

            ++stats.nonClearCount;
            if (isRedDominant(pixel))
            {
                ++stats.redDominantCount;
            }
            if (isGreenDominant(pixel))
            {
                ++stats.greenDominantCount;
            }
        }
    }

    return stats;
}

std::string withSuffixBeforeExtension(std::string path, const std::string& suffix)
{
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
    {
        return path + suffix;
    }
    path.insert(dot, suffix);
    return path;
}

bool writePpm(const std::string& path, const RenderTargetReadbackEvent& event)
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
        const std::uint8_t* src = event.colorBytes.data() + static_cast<std::size_t>(y) * event.rowStrideBytes;
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

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "CubeDepth.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Vec3f& normal, const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, const Vec3f& v3) {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 1u);
        mesh.indices.push_back(baseIndex + 2u);
        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 2u);
        mesh.indices.push_back(baseIndex + 3u);
    };

    const float h = halfExtent;

    addFace({0.0f, 0.0f, 1.0f}, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h});
    addFace({0.0f, 0.0f, -1.0f}, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h});
    addFace({-1.0f, 0.0f, 0.0f}, {-h, -h, -h}, {-h, -h, h}, {-h, h, h}, {-h, h, -h});
    addFace({1.0f, 0.0f, 0.0f}, {h, -h, h}, {h, -h, -h}, {h, h, -h}, {h, h, h});
    addFace({0.0f, 1.0f, 0.0f}, {-h, h, h}, {h, h, h}, {h, h, -h}, {-h, h, -h});
    addFace({0.0f, -1.0f, 0.0f}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h});

    return mesh;
}

} // namespace

int main(int argc, char** argv)
{
    RuntimeConfig config{};
    config.graphicsDeviceDesc.preferredBackend = GraphicsBackend::Vulkan;
    config.graphicsDeviceDesc.enableValidation = false;

    std::uint64_t numFrames = 2;
    std::string outputPath = "cube_depth_readback.ppm";

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

    if (numFrames < 2)
    {
        numFrames = 2;
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
    targetDesc.width = 640;
    targetDesc.height = 480;
    targetDesc.cpuReadback = true;
    targetDesc.debugName = "CubeDepth.Target";
    const RenderTargetHandle target = graphicsDevice->createRenderTarget(targetDesc);
    if (!graphicsDevice->isValidRenderTarget(target))
    {
        runtime.shutdown();
        std::cerr << "Failed to create readback target.\n";
        return 1;
    }

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.1f, 4.2f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 52.0f;
    camera.outputTarget = target;
    camera.outputWidth = targetDesc.width;
    camera.outputHeight = targetDesc.height;
    camera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.35f, -0.45f, -1.0f};
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 4.0f;
    world.setDirectionalLight(lightEntity, light);

    auto& resources = runtime.getScene().resources();
    const auto cubeMesh = resources.registerMesh(makeCubeMesh(0.65f));

    MaterialResourceDesc frontMaterialDesc{};
    frontMaterialDesc.debugName = "CubeDepth.FrontMaterial";
    frontMaterialDesc.baseColor = {0.95f, 0.10f, 0.08f};
    frontMaterialDesc.metallic = 0.0f;
    frontMaterialDesc.roughness = 0.45f;
    const auto frontMaterial = resources.registerMaterial(frontMaterialDesc);

    MaterialResourceDesc backMaterialDesc{};
    backMaterialDesc.debugName = "CubeDepth.BackMaterial";
    backMaterialDesc.baseColor = {0.10f, 0.85f, 0.12f};
    backMaterialDesc.metallic = 0.0f;
    backMaterialDesc.roughness = 0.45f;
    const auto backMaterial = resources.registerMaterial(backMaterialDesc);

    const auto frontCubeEntity = world.createEntity();
    TransformComponent frontCubeTransform{};
    frontCubeTransform.worldTransform.position = {0.18f, -0.02f, 0.05f};
    frontCubeTransform.worldTransform.rotation = quaternionFromEulerDegrees(-18.0f, 32.0f, 0.0f);
    world.setTransform(frontCubeEntity, frontCubeTransform);
    MeshRendererComponent frontCube{};
    frontCube.mesh = cubeMesh;
    frontCube.material = frontMaterial;
    // Start hidden to capture a back-only frame, then enable to verify depth at overlap.
    frontCube.visible = false;
    world.setMeshRenderer(frontCubeEntity, frontCube);

    // Intentionally created after the front cube so depth testing, not draw order, resolves visibility.
    const auto backCubeEntity = world.createEntity();
    TransformComponent backCubeTransform{};
    backCubeTransform.worldTransform.position = {-0.14f, 0.03f, -1.35f};
    backCubeTransform.worldTransform.rotation = quaternionFromEulerDegrees(12.0f, -24.0f, 0.0f);
    backCubeTransform.worldTransform.scale = {1.35f, 1.35f, 1.35f};
    world.setTransform(backCubeEntity, backCubeTransform);
    MeshRendererComponent backCube{};
    backCube.mesh = cubeMesh;
    backCube.material = backMaterial;
    backCube.visible = true;
    world.setMeshRenderer(backCubeEntity, backCube);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    std::vector<RenderTargetReadbackRequest> readbackRequests;
    for (std::uint64_t i = 0; i < numFrames; ++i)
    {
        if (i == 1)
        {
            frontCube.visible = true;
            world.setMeshRenderer(frontCubeEntity, frontCube);
        }

        const RenderTargetReadbackRequest request = graphicsDevice->requestRenderTargetReadback(target);
        if (request.id != 0)
        {
            readbackRequests.push_back(request);
        }

        frame.frameIndex = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.tick(frame);
    }

    RenderTargetReadbackEvent backOnlyCapture{};
    RenderTargetReadbackEvent layeredCapture{};
    bool hasBackOnlyPayload = false;
    bool hasLayeredPayload = false;
    for (const RenderTargetReadbackRequest request : readbackRequests)
    {
        RenderTargetReadbackEvent event{};
        if (!graphicsDevice->tryGetRenderTargetReadback(request, event))
        {
            continue;
        }
        if (event.target.id == target.id && !event.colorBytes.empty() && isValidReadback(event))
        {
            if (!hasBackOnlyPayload || event.frameIndex < backOnlyCapture.frameIndex)
            {
                backOnlyCapture = event;
                hasBackOnlyPayload = true;
            }
            if (!hasLayeredPayload || event.frameIndex >= layeredCapture.frameIndex)
            {
                layeredCapture = event;
                hasLayeredPayload = true;
            }
        }
    }

    runtime.shutdown();

    if (config.graphicsDeviceDesc.preferredBackend == GraphicsBackend::Null)
    {
        std::cout << "Null backend selected; depth capture skipped.\n";
        return 0;
    }

    if (!hasBackOnlyPayload || !hasLayeredPayload)
    {
        std::cerr << "Expected two readback payloads (back-only and layered), but did not receive them.\n";
        return 1;
    }

    if (!containsNonClearPixel(layeredCapture))
    {
        std::cerr << "Captured image appears to contain only clear color.\n";
        return 1;
    }

    const auto backOnlyCenter = readCenterPixel(backOnlyCapture);
    const auto layeredCenter = readCenterPixel(layeredCapture);
    if (!isGreenDominant(backOnlyCenter))
    {
        std::cerr << "Back-only frame center pixel expected green dominance. Observed RGBA=("
                  << static_cast<unsigned>(backOnlyCenter[0]) << ", " << static_cast<unsigned>(backOnlyCenter[1]) << ", "
                  << static_cast<unsigned>(backOnlyCenter[2]) << ", " << static_cast<unsigned>(backOnlyCenter[3]) << ")\n";
        return 1;
    }
    if (!isRedDominant(layeredCenter))
    {
        std::cerr << "Layered frame center pixel expected red dominance from near cube. Observed RGBA=("
                  << static_cast<unsigned>(layeredCenter[0]) << ", " << static_cast<unsigned>(layeredCenter[1]) << ", "
                  << static_cast<unsigned>(layeredCenter[2]) << ", " << static_cast<unsigned>(layeredCenter[3]) << ")\n";
        return 1;
    }

    const DominantPixelStats layeredStats = analyzeDominantPixels(layeredCapture);
    if (layeredStats.redDominantCount < 200u || layeredStats.greenDominantCount < 80u)
    {
        std::cerr << "Layered frame expected substantial red and green regions, but counts were red="
                  << layeredStats.redDominantCount << ", green=" << layeredStats.greenDominantCount
                  << ", non-clear=" << layeredStats.nonClearCount << '\n';
        return 1;
    }

    if (!writePpm(outputPath, layeredCapture))
    {
        std::cerr << "Failed to write image: " << outputPath << '\n';
        return 1;
    }
    const std::string backOnlyPath = withSuffixBeforeExtension(outputPath, "_back_only");
    if (!writePpm(backOnlyPath, backOnlyCapture))
    {
        std::cerr << "Failed to write image: " << backOnlyPath << '\n';
        return 1;
    }

    std::cout << "Cube depth capture passed. back-only center RGBA=(" << static_cast<unsigned>(backOnlyCenter[0]) << ", "
              << static_cast<unsigned>(backOnlyCenter[1]) << ", " << static_cast<unsigned>(backOnlyCenter[2]) << ", "
              << static_cast<unsigned>(backOnlyCenter[3]) << "), layered center RGBA=("
              << static_cast<unsigned>(layeredCenter[0]) << ", " << static_cast<unsigned>(layeredCenter[1]) << ", "
              << static_cast<unsigned>(layeredCenter[2]) << ", " << static_cast<unsigned>(layeredCenter[3]) << "), "
              << "layered dominant counts red=" << layeredStats.redDominantCount << ", green=" << layeredStats.greenDominantCount << ", "
              << "wrote layered image to " << outputPath << " and back-only image to " << backOnlyPath << '\n';
    return 0;
}
