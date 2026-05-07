#include "common/frame_context.h"
#include "common/logger.h"
#include "helpers/readback.h"
#include "engine/components.h"
#include "engine/runtime.h"


#include <cmath>
#include <cstdint>
#include <cstdlib>
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
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::tests::helpers::ReadbackPixel;

constexpr float kClearTolerance = 0.02f;
constexpr std::uint32_t kCenterWindowRadius = 16u;

void printUsage(const char* appName)
{
    CRESSIM_LOG_ERROR( "Usage: " , appName , " [--output path.ppm]\n");
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

bool isNear(ReadbackPixel value, const ReadbackPixel& expected, float tolerance)
{
    return std::fabs(value.r - expected.r) <= tolerance &&
           std::fabs(value.g - expected.g) <= tolerance &&
           std::fabs(value.b - expected.b) <= tolerance &&
           std::fabs(value.a - expected.a) <= tolerance;
}

bool isClearPixel(const ReadbackPixel& pixel, const ReadbackPixel& clearPixel)
{
    return isNear(pixel, clearPixel, kClearTolerance);
}

bool containsNonClearPixel(const GpuRenderTargetReadbackEvent& event, const ReadbackPixel& clearPixel)
{
    if (!cressim::neo::tests::helpers::isValidReadback(event))
    {
        return false;
    }

    for (std::uint32_t y = 0; y < event.height; ++y)
    {
        for (std::uint32_t x = 0; x < event.width; ++x)
        {
            if (!isClearPixel(cressim::neo::tests::helpers::decodePixel(event, x, y), clearPixel))
            {
                return true;
            }
        }
    }

    return false;
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

struct PixelRect
{
    std::uint32_t startX = 0;
    std::uint32_t startY = 0;
    std::uint32_t endX = 0;
    std::uint32_t endY = 0;
};

PixelRect makeCenterWindowRect(const GpuRenderTargetReadbackEvent& event)
{
    const std::uint32_t centerX = event.width / 2u;
    const std::uint32_t centerY = event.height / 2u;
    return {
        centerX > kCenterWindowRadius ? centerX - kCenterWindowRadius : 0u,
        centerY > kCenterWindowRadius ? centerY - kCenterWindowRadius : 0u,
        centerX + kCenterWindowRadius + 1u,
        centerY + kCenterWindowRadius + 1u};
}

DominantPixelStats analyzeDominantPixelsInRect(const GpuRenderTargetReadbackEvent& event,
                                               const PixelRect& rect,
                                               const ReadbackPixel& clearPixel)
{
    DominantPixelStats stats{};
    if (!cressim::neo::tests::helpers::isValidReadback(event))
    {
        return stats;
    }

    const std::uint32_t clampedStartX = std::min(rect.startX, event.width);
    const std::uint32_t clampedStartY = std::min(rect.startY, event.height);
    const std::uint32_t clampedEndX = std::min(rect.endX, event.width);
    const std::uint32_t clampedEndY = std::min(rect.endY, event.height);

    for (std::uint32_t y = clampedStartY; y < clampedEndY; ++y)
    {
        for (std::uint32_t x = clampedStartX; x < clampedEndX; ++x)
        {
            const ReadbackPixel pixel = cressim::neo::tests::helpers::decodePixel(event, x, y);
            if (isClearPixel(pixel, clearPixel))
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
    return analyzeDominantPixelsInRect(event, PixelRect{0u, 0u, event.width, event.height},
                                       clearPixel);
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
    config.gpuDeviceDesc.preferredBackend = cressim::neo::gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    constexpr std::uint64_t kNumFrames = 2u;
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
    for (std::uint64_t i = 0; i < kNumFrames; ++i)
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
        if (event.binding.target.id == target.id && !event.colorBytes.empty() &&
            cressim::neo::tests::helpers::isValidReadback(event))
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

    const auto backOnlyCenter = cressim::neo::tests::helpers::readCenterPixel(backOnlyCapture);
    const auto layeredCenter = cressim::neo::tests::helpers::readCenterPixel(layeredCapture);
    const PixelRect centerWindow = makeCenterWindowRect(backOnlyCapture);
    const DominantPixelStats backOnlyCenterStats =
        analyzeDominantPixelsInRect(backOnlyCapture, centerWindow, clearPixel);
    const DominantPixelStats layeredCenterStats =
        analyzeDominantPixelsInRect(layeredCapture, centerWindow, clearPixel);

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

    if (!outputPath.empty() &&
        !cressim::neo::tests::helpers::writePpm(outputPath, layeredCapture))
    {
        CRESSIM_LOG_ERROR( "Failed to write image: " , outputPath , '\n');
        return 1;
    }
    std::string backOnlyPath;
    if (!outputPath.empty())
    {
        backOnlyPath = withSuffixBeforeExtension(outputPath, "_back_only");
        if (!cressim::neo::tests::helpers::writePpm(backOnlyPath, backOnlyCapture))
        {
            CRESSIM_LOG_ERROR( "Failed to write image: " , backOnlyPath , '\n');
            return 1;
        }
    }

    CRESSIM_LOG_INFO( "Cube depth capture passed. back-only center RGBA=(" , backOnlyCenter.r , ", "
              , backOnlyCenter.g , ", " , backOnlyCenter.b , ", "
              , backOnlyCenter.a , "), layered center RGBA=("
              , layeredCenter.r , ", " , layeredCenter.g , ", "
              , layeredCenter.b , ", " , layeredCenter.a , "), "
              , "layered dominant counts red=" , layeredStats.redDominantCount , ", green="
              , layeredStats.greenDominantCount);
    if (!outputPath.empty())
    {
        CRESSIM_LOG_INFO( ", wrote layered image to " , outputPath , " and back-only image to "
                  , backOnlyPath , '\n');
    }
    else
    {
        CRESSIM_LOG_INFO( ".\n");
    }
    return 0;
}
