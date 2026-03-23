#include "common/frame_context.h"
#include "common/id.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "common/logger.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    CRESSIM_LOG_ERROR( "Usage: " , appName , " [--backend vulkan|null] [--frames N] [--validation on|off]\n");
}

bool isNear(std::uint8_t value, std::uint8_t expected, std::uint8_t tolerance)
{
    const int diff = static_cast<int>(value) - static_cast<int>(expected);
    return diff >= -static_cast<int>(tolerance) && diff <= static_cast<int>(tolerance);
}

bool containsNonClearPixel(const GpuRenderTargetReadbackEvent& event)
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
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

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

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
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

    GpuDevice* graphicsDevice = runtime.getGpuDevice();
    GpuRenderTargetHandle secondaryTarget{};
    if (graphicsDevice != nullptr)
    {
        GpuRenderTargetDesc secondaryTargetDesc{};
        secondaryTargetDesc.width = 640;
        secondaryTargetDesc.height = 480;
        secondaryTargetDesc.arraySize = 2u;
        secondaryTargetDesc.layeredRendering = true;
        secondaryTargetDesc.debugName = "Smoke.SecondaryCamera";
        secondaryTarget = graphicsDevice->renderTargetSystem().createRenderTarget(secondaryTargetDesc);
    }

    if (graphicsDevice != nullptr && graphicsDevice->renderTargetSystem().isValidRenderTarget(secondaryTarget))
    {
        const auto secondaryCameraEntity = world.createEntity();
        TransformComponent secondaryCameraTransform{};
        secondaryCameraTransform.worldTransform.position = {-1.0f, 1.5f, -2.5f};

        CameraComponent secondaryCamera{};
        secondaryCamera.output.mode = cressim::neo::gpu::CameraOutputMode::ExplicitSurface;
        secondaryCamera.output.binding = cressim::neo::gpu::GpuRenderTargetBinding{secondaryTarget, 0u, 1u};
        secondaryCamera.outputWidth = 800;
        secondaryCamera.outputHeight = 600;
        secondaryCamera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
        secondaryCamera.renderOrder = 1;

        world.setTransform(secondaryCameraEntity, secondaryCameraTransform);
        world.setCamera(secondaryCameraEntity, secondaryCamera);
    }

    cressim::neo::common::EntityId extraSecondaryCameraEntity =
        cressim::neo::common::kInvalidEntityId;

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

    const auto rigidEntity = world.createEntity();
    TransformComponent rigidTransform{};
    rigidTransform.worldTransform.position = {-0.2f, 0.1f, 0.0f};
    world.setTransform(rigidEntity, rigidTransform);
    cressim::neo::engine::RigidBodyComponent rigidBody{};
    rigidBody.simulated = true;
    rigidBody.inverseMass = 1.0f;
    rigidBody.linearVelocity = {0.25f, 0.0f, 0.0f};
    rigidBody.angularVelocity = {0.0f, 0.1f, 0.0f};
    world.setRigidBody(rigidEntity, rigidBody);
    cressim::neo::engine::ColliderComponent rigidCollider{};
    rigidCollider.shapeType = cressim::neo::physics::ColliderShapeType::Sphere;
    rigidCollider.shapeParams = {0.4f, 0.0f, 0.0f, 0.0f};
    world.addCollider(rigidEntity, rigidCollider);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    std::vector<GpuRenderTargetReadbackRequest> readbackRequests;

    for (std::uint64_t i = 0; i < numFrames; ++i)
    {
        if (graphicsDevice != nullptr && graphicsDevice->renderTargetSystem().isValidRenderTarget(secondaryTarget))
        {
            const GpuRenderTargetReadbackRequest request =
                graphicsDevice->renderTargetSystem().requestRenderTargetReadback(
                    cressim::neo::gpu::GpuRenderTargetBinding{secondaryTarget, 0u, 1u});
            if (request.id != 0)
            {
                readbackRequests.push_back(request);
            }
            if (extraSecondaryCameraEntity != cressim::neo::common::kInvalidEntityId)
            {
                const GpuRenderTargetReadbackRequest layerOneRequest =
                    graphicsDevice->renderTargetSystem().requestRenderTargetReadback(
                        cressim::neo::gpu::GpuRenderTargetBinding{secondaryTarget, 1u, 1u});
                if (layerOneRequest.id != 0)
                {
                    readbackRequests.push_back(layerOneRequest);
                }
            }
        }

        if (i == 1u && graphicsDevice != nullptr &&
            graphicsDevice->renderTargetSystem().isValidRenderTarget(secondaryTarget) &&
            extraSecondaryCameraEntity == cressim::neo::common::kInvalidEntityId)
        {
            extraSecondaryCameraEntity = world.createEntity();
            TransformComponent extraCameraTransform{};
            extraCameraTransform.worldTransform.position = {1.0f, 1.2f, -2.8f};

            CameraComponent extraCamera{};
            extraCamera.output.mode = cressim::neo::gpu::CameraOutputMode::ExplicitSurface;
            extraCamera.output.binding =
                cressim::neo::gpu::GpuRenderTargetBinding{secondaryTarget, 1u, 1u};
            extraCamera.outputWidth = 800;
            extraCamera.outputHeight = 600;
            extraCamera.viewport = {0.0f, 0.0f, 1.0f, 1.0f};
            extraCamera.renderOrder = 2;

            world.setTransform(extraSecondaryCameraEntity, extraCameraTransform);
            world.setCamera(extraSecondaryCameraEntity, extraCamera);
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
        for (const GpuRenderTargetReadbackRequest request : readbackRequests)
        {
            GpuRenderTargetReadbackEvent event{};
            if (!graphicsDevice->renderTargetSystem().tryGetRenderTargetReadback(request, event))
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

    if (config.gpuDeviceDesc.preferredBackend == GpuBackend::Vulkan)
    {
        if (readbackEvents == 0)
        {
            runtime.shutdown();
            CRESSIM_LOG_ERROR( "Smoke run failed: expected at least one readback event.\n");
            return 1;
        }
        if (payloadEvents == 0)
        {
            runtime.shutdown();
            CRESSIM_LOG_ERROR( "Smoke run failed: expected readback payload for Vulkan backend.\n");
            return 1;
        }
        if (!foundNonClearPixel)
        {
            runtime.shutdown();
            CRESSIM_LOG_ERROR( "Smoke run failed: readback payload contains only clear color.\n");
            return 1;
        }
    }

    runtime.shutdown();

    CRESSIM_LOG_INFO( "Smoke run passed. Frames: " , numFrames , ", Readback events: " , readbackEvents
              , ", Payload events: " , payloadEvents , '\n');
    return 0;
}
