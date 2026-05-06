#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "common/logger.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsAccessories/interface/GraphicsAccessories.hpp"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;

GpuBackend parseBackend(const std::string& value)
{
    if (value == "null")
    {
        return GpuBackend::Null;
    }
    if (value == "vulkan")
    {
        return GpuBackend::Vulkan;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

void printUsage(const char* appName)
{
    CRESSIM_LOG_ERROR( "Usage: " , appName
              , " [--backend vulkan|null] [--frames N] [--output path.ppm] [--validation on|off]\n");
}

float degreesToRadians(float value)
{
    return value * 0.017453292519943295769f;
}

Diligent::QuaternionF quaternionFromEulerDegrees(float pitchDegrees, float yawDegrees, float rollDegrees)
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

    return Diligent::QuaternionF{
        sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
        cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
        cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
        cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw};
}

bool isValidReadback(const GpuRenderTargetReadbackEvent& event)
{
    if (event.width == 0 || event.height == 0)
    {
        return false;
    }

    const auto& formatAttribs = Diligent::GetTextureFormatAttribs(event.colorFormat);
    if (formatAttribs.Format == Diligent::TEX_FORMAT_UNKNOWN || formatAttribs.IsTypeless ||
        formatAttribs.ComponentType == Diligent::COMPONENT_TYPE_COMPRESSED)
    {
        return false;
    }

    const std::uint32_t minStride = event.width * formatAttribs.GetElementSize();
    if (event.rowStrideBytes < minStride)
    {
        return false;
    }

    if (event.colorBytes.size() <
        static_cast<std::size_t>(event.rowStrideBytes) * static_cast<std::size_t>(event.height))
    {
        return false;
    }
    return true;
}

struct ReadbackPixel
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};

float halfToFloat(std::uint16_t value)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    std::uint32_t exponent   = (value >> 10u) & 0x1fu;
    std::uint32_t mantissa   = value & 0x03ffu;

    std::uint32_t bits = 0u;
    if (exponent == 0u)
    {
        if (mantissa != 0u)
        {
            exponent = 113u;
            while ((mantissa & 0x0400u) == 0u)
            {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (exponent << 23u) | (mantissa << 13u);
        }
        else
        {
            bits = sign;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::size_t pixelOffset(const GpuRenderTargetReadbackEvent& event, std::uint32_t x, std::uint32_t y)
{
    return static_cast<std::size_t>(y) * event.rowStrideBytes +
           static_cast<std::size_t>(x) *
               Diligent::GetTextureFormatAttribs(event.colorFormat).GetElementSize();
}

ReadbackPixel decodePixel(const GpuRenderTargetReadbackEvent& event, std::uint32_t x, std::uint32_t y)
{
    ReadbackPixel pixel{};
    if (!isValidReadback(event))
    {
        return pixel;
    }

    const std::size_t offset = pixelOffset(event, x, y);
    if (event.colorFormat == Diligent::TEX_FORMAT_RGBA16_FLOAT)
    {
        const std::uint16_t pixelR =
            static_cast<std::uint16_t>(event.colorBytes[offset + 0u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 1u]) << 8u);
        const std::uint16_t pixelG =
            static_cast<std::uint16_t>(event.colorBytes[offset + 2u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 3u]) << 8u);
        const std::uint16_t pixelB =
            static_cast<std::uint16_t>(event.colorBytes[offset + 4u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 5u]) << 8u);
        const std::uint16_t pixelA =
            static_cast<std::uint16_t>(event.colorBytes[offset + 6u]) |
            (static_cast<std::uint16_t>(event.colorBytes[offset + 7u]) << 8u);
        pixel.r = halfToFloat(pixelR);
        pixel.g = halfToFloat(pixelG);
        pixel.b = halfToFloat(pixelB);
        pixel.a = halfToFloat(pixelA);
        return pixel;
    }

    pixel.r = static_cast<float>(event.colorBytes[offset + 0u]) / 255.0f;
    pixel.g = static_cast<float>(event.colorBytes[offset + 1u]) / 255.0f;
    pixel.b = static_cast<float>(event.colorBytes[offset + 2u]) / 255.0f;
    pixel.a = static_cast<float>(event.colorBytes[offset + 3u]) / 255.0f;
    return pixel;
}

std::uint8_t encodeByte(float value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

bool isNear(ReadbackPixel value, const ReadbackPixel& expected, float tolerance)
{
    return std::fabs(value.r - expected.r) <= tolerance &&
           std::fabs(value.g - expected.g) <= tolerance &&
           std::fabs(value.b - expected.b) <= tolerance &&
           std::fabs(value.a - expected.a) <= tolerance;
}

bool containsNonClearPixel(const GpuRenderTargetReadbackEvent& event, const ReadbackPixel& clearPixel)
{
    constexpr float kTolerance = 0.02f;

    if (!isValidReadback(event))
    {
        return false;
    }

    for (std::uint32_t y = 0; y < event.height; ++y)
    {
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            if (!isNear(decodePixel(event, x, y), clearPixel, kTolerance))
            {
                return true;
            }
        }
    }

    return false;
}

ReadbackPixel readCenterPixel(const GpuRenderTargetReadbackEvent& event)
{
    ReadbackPixel pixel{};
    if (!isValidReadback(event))
    {
        return pixel;
    }

    const std::uint32_t x = event.width / 2u;
    const std::uint32_t y = event.height / 2u;
    return decodePixel(event, x, y);
}

bool isRedDominant(const ReadbackPixel& pixel)
{
    constexpr float kDominance = 0.05f;
    return pixel.r > pixel.g + kDominance && pixel.r > pixel.b + kDominance;
}

bool isGreenDominant(const ReadbackPixel& pixel)
{
    constexpr float kDominance = 0.05f;
    return pixel.g > pixel.r + kDominance && pixel.g > pixel.b + kDominance;
}

struct DominantPixelStats
{
    std::uint64_t nonClearCount = 0;
    std::uint64_t redDominantCount = 0;
    std::uint64_t greenDominantCount = 0;
};

DominantPixelStats analyzeDominantPixelsInRect(const GpuRenderTargetReadbackEvent& event,
                                               std::uint32_t startX, std::uint32_t startY,
                                               std::uint32_t endX, std::uint32_t endY,
                                               const ReadbackPixel& clearPixel)
{
    DominantPixelStats stats{};
    if (!isValidReadback(event))
    {
        return stats;
    }
    constexpr float kTolerance = 0.02f;

    const std::uint32_t clampedStartX = std::min(startX, event.width);
    const std::uint32_t clampedStartY = std::min(startY, event.height);
    const std::uint32_t clampedEndX = std::min(endX, event.width);
    const std::uint32_t clampedEndY = std::min(endY, event.height);

    for (std::uint32_t y = clampedStartY; y < clampedEndY; ++y)
    {
        for (std::uint32_t x = clampedStartX; x < clampedEndX; ++x)
        {
            const ReadbackPixel pixel = decodePixel(event, x, y);
            if (isNear(pixel, clearPixel, kTolerance))
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

DominantPixelStats analyzeDominantPixels(const GpuRenderTargetReadbackEvent& event,
                                         const ReadbackPixel& clearPixel)
{
    return analyzeDominantPixelsInRect(event, 0u, 0u, event.width, event.height, clearPixel);
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
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            const ReadbackPixel pixel = decodePixel(event, x, y);
            rgbRow[static_cast<std::size_t>(x) * 3u + 0u] = encodeByte(pixel.r);
            rgbRow[static_cast<std::size_t>(x) * 3u + 1u] = encodeByte(pixel.g);
            rgbRow[static_cast<std::size_t>(x) * 3u + 2u] = encodeByte(pixel.b);
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

    const auto addFace = [&](const Diligent::float3& normal, const Diligent::float3& v0, const Diligent::float3& v1, const Diligent::float3& v2, const Diligent::float3& v3) {
        const std::uint32_t baseIndex = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f});
        mesh.vertices.push_back({v1, normal, 1.0f, 0.0f});
        mesh.vertices.push_back({v2, normal, 1.0f, 1.0f});
        mesh.vertices.push_back({v3, normal, 0.0f, 1.0f});

        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 2u);
        mesh.indices.push_back(baseIndex + 1u);
        mesh.indices.push_back(baseIndex + 0u);
        mesh.indices.push_back(baseIndex + 3u);
        mesh.indices.push_back(baseIndex + 2u);
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
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

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
            config.gpuDeviceDesc.preferredBackend = parseBackend(argv[++i]);
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
                config.gpuDeviceDesc.enableValidation = true;
                continue;
            }
            if (value == "off")
            {
                config.gpuDeviceDesc.enableValidation = false;
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
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    auto& world = runtime.getWorld();
    auto* graphicsDevice = runtime.getGpuDevice();
    if (graphicsDevice == nullptr)
    {
        runtime.shutdown();
        CRESSIM_LOG_ERROR( "Graphics device unavailable.\n");
        return 1;
    }

    GpuRenderTargetDesc targetDesc{};
    targetDesc.width = 640;
    targetDesc.height = 480;
    targetDesc.debugName = "CubeDepth.Target";
    const GpuRenderTargetHandle target = graphicsDevice->renderTargetSystem().createRenderTarget(targetDesc);
    if (!graphicsDevice->renderTargetSystem().isValidRenderTarget(target))
    {
        runtime.shutdown();
        CRESSIM_LOG_ERROR( "Failed to create readback target.\n");
        return 1;
    }

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.1f, -4.2f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.verticalFovDegrees = 52.0f;
    camera.output.mode = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
    camera.output.binding = cressim::neo::gpu::GpuRenderTargetBinding{target, 0u, 1u};
    camera.outputWidth = targetDesc.width;
    camera.outputHeight = targetDesc.height;
    camera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
    camera.clearColorValue = {0.02f, 0.02f, 0.03f, 1.0f};
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    DirectionalLightComponent light{};
    light.direction = {-0.35f, -0.45f, 1.0f};
    light.color = {1.0f, 1.0f, 1.0f};
    light.intensity = 4.0f;
    world.setDirectionalLight(lightEntity, light);

    auto& resources = runtime.getResources();
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
    frontCubeTransform.worldTransform.position = {0.18f, -0.02f, -0.05f};
    frontCubeTransform.worldTransform.rotation = quaternionFromEulerDegrees(-18.0f, 32.0f, 0.0f);
    world.setTransform(frontCubeEntity, frontCubeTransform);
    MeshRendererComponent frontCube{};
    frontCube.mesh = cubeMesh;
    frontCube.material = frontMaterial;
    // Start hidden to capture a back-only frame, then enable the front cube to verify depth at overlap.
    frontCube.visible = false;
    world.setMeshRenderer(frontCubeEntity, frontCube);

    // Intentionally created after the front cube so depth testing, not draw order, resolves visibility.
    const auto backCubeEntity = world.createEntity();
    TransformComponent backCubeTransform{};
    backCubeTransform.worldTransform.position = {-0.14f, 0.03f, 1.35f};
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
    std::vector<GpuRenderTargetReadbackRequest> readbackRequests;
    for (std::uint64_t i = 0; i < numFrames; ++i)
    {
        if (i == 1)
        {
            frontCube.visible = true;
            world.setMeshRenderer(frontCubeEntity, frontCube);
        }

        const GpuRenderTargetReadbackRequest request =
            graphicsDevice->renderTargetSystem().requestRenderTargetReadback(
                cressim::neo::gpu::GpuRenderTargetBinding{target, 0u, 1u});
        if (request.id != 0)
        {
            readbackRequests.push_back(request);
        }

        frame.frameIndex = i;
        frame.timeSeconds = static_cast<double>(i) * static_cast<double>(frame.deltaSeconds);
        runtime.tick(frame);
    }

    GpuRenderTargetReadbackEvent backOnlyCapture{};
    GpuRenderTargetReadbackEvent layeredCapture{};
    bool hasBackOnlyPayload = false;
    bool hasLayeredPayload = false;
    for (const GpuRenderTargetReadbackRequest request : readbackRequests)
    {
        GpuRenderTargetReadbackEvent event{};
        if (!graphicsDevice->renderTargetSystem().tryGetRenderTargetReadback(request, event))
        {
            continue;
        }
        if (event.binding.target.id == target.id && !event.colorBytes.empty() && isValidReadback(event))
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

    if (config.gpuDeviceDesc.preferredBackend == GpuBackend::Null)
    {
        CRESSIM_LOG_INFO("Null backend selected; depth capture skipped.\n");
        return 0;
    }

    if (!hasBackOnlyPayload || !hasLayeredPayload)
    {
        CRESSIM_LOG_ERROR( "Expected two readback payloads (back-only and layered), but did not receive them.\n");
        return 1;
    }

    const ReadbackPixel clearPixel{camera.clearColorValue.x, camera.clearColorValue.y,
                                   camera.clearColorValue.z, camera.clearColorValue.w};

    if (!containsNonClearPixel(layeredCapture, clearPixel))
    {
        CRESSIM_LOG_ERROR( "Captured image appears to contain only clear color.\n");
        return 1;
    }

    const auto backOnlyCenter = readCenterPixel(backOnlyCapture);
    const auto layeredCenter = readCenterPixel(layeredCapture);
    const std::uint32_t centerX = backOnlyCapture.width / 2u;
    const std::uint32_t centerY = backOnlyCapture.height / 2u;
    constexpr std::uint32_t kCenterWindowRadius = 16u;
    const DominantPixelStats backOnlyCenterStats =
        analyzeDominantPixelsInRect(backOnlyCapture,
                                    centerX > kCenterWindowRadius ? centerX - kCenterWindowRadius : 0u,
                                    centerY > kCenterWindowRadius ? centerY - kCenterWindowRadius : 0u,
                                    centerX + kCenterWindowRadius + 1u,
                                    centerY + kCenterWindowRadius + 1u,
                                    clearPixel);
    const DominantPixelStats layeredCenterStats =
        analyzeDominantPixelsInRect(layeredCapture,
                                    centerX > kCenterWindowRadius ? centerX - kCenterWindowRadius : 0u,
                                    centerY > kCenterWindowRadius ? centerY - kCenterWindowRadius : 0u,
                                    centerX + kCenterWindowRadius + 1u,
                                    centerY + kCenterWindowRadius + 1u,
                                    clearPixel);

    if (backOnlyCenterStats.greenDominantCount == 0u ||
        backOnlyCenterStats.greenDominantCount <= backOnlyCenterStats.redDominantCount)
    {
        CRESSIM_LOG_ERROR( "Back-only frame center window expected green dominance. Center RGBA=("
                  , backOnlyCenter.r , ", " , backOnlyCenter.g , ", "
                  , backOnlyCenter.b , ", " , backOnlyCenter.a
                  , "), center-window red=" , backOnlyCenterStats.redDominantCount
                  , ", green=" , backOnlyCenterStats.greenDominantCount
                  , ", non-clear=" , backOnlyCenterStats.nonClearCount , '\n');
        return 1;
    }
    if (layeredCenterStats.redDominantCount == 0u ||
        layeredCenterStats.redDominantCount <= layeredCenterStats.greenDominantCount)
    {
        CRESSIM_LOG_ERROR( "Layered frame center window expected red dominance from near cube. Center RGBA=("
                  , layeredCenter.r , ", " , layeredCenter.g , ", "
                  , layeredCenter.b , ", " , layeredCenter.a
                  , "), center-window red=" , layeredCenterStats.redDominantCount
                  , ", green=" , layeredCenterStats.greenDominantCount
                  , ", non-clear=" , layeredCenterStats.nonClearCount , '\n');
        return 1;
    }

    const DominantPixelStats layeredStats = analyzeDominantPixels(layeredCapture, clearPixel);
    if (layeredStats.redDominantCount < 200u || layeredStats.greenDominantCount < 80u)
    {
        CRESSIM_LOG_ERROR( "Layered frame expected substantial red and green regions, but counts were red="
                  , layeredStats.redDominantCount , ", green=" , layeredStats.greenDominantCount
                  , ", non-clear=" , layeredStats.nonClearCount , '\n');
        return 1;
    }

    if (!writePpm(outputPath, layeredCapture))
    {
        CRESSIM_LOG_ERROR( "Failed to write image: " , outputPath , '\n');
        return 1;
    }
    const std::string backOnlyPath = withSuffixBeforeExtension(outputPath, "_back_only");
    if (!writePpm(backOnlyPath, backOnlyCapture))
    {
        CRESSIM_LOG_ERROR( "Failed to write image: " , backOnlyPath , '\n');
        return 1;
    }

    CRESSIM_LOG_INFO( "Cube depth capture passed. back-only center RGBA=(" , backOnlyCenter.r , ", "
              , backOnlyCenter.g , ", " , backOnlyCenter.b , ", "
              , backOnlyCenter.a , "), layered center RGBA=("
              , layeredCenter.r , ", " , layeredCenter.g , ", "
              , layeredCenter.b , ", " , layeredCenter.a , "), "
              , "layered dominant counts red=" , layeredStats.redDominantCount , ", green=" , layeredStats.greenDominantCount , ", "
              , "wrote layered image to " , outputPath , " and back-only image to " , backOnlyPath , '\n');
    return 0;
}
