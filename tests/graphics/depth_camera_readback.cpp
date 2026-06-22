#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/readback.h"

#include <cmath>
#include <cstdint>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::gpu::GpuRenderTargetDepthReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetDepthReadbackRequest;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;

float degreesToRadians(float value)
{
    return value * 0.017453292519943295769f;
}

Diligent::QuaternionF quaternionFromEulerDegrees(float pitchDegrees, float yawDegrees,
                                                 float rollDegrees)
{
    const float pitch = degreesToRadians(pitchDegrees) * 0.5f;
    const float yaw   = degreesToRadians(yawDegrees) * 0.5f;
    const float roll  = degreesToRadians(rollDegrees) * 0.5f;

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

MeshResourceDesc makeCubeMesh(float halfExtent)
{
    MeshResourceDesc mesh{};
    mesh.debugName = "DepthCamera.CubeMesh";
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3 &normal, const Diligent::float3 &v0,
                             const Diligent::float3 &v1, const Diligent::float3 &v2,
                             const Diligent::float3 &v3)
    {
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

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = cressim::neo::gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping depth camera readback test because runtime initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();
    auto *graphicsDevice = runtime.getGpuDevice();
    if (graphicsDevice == nullptr)
    {
        CRESSIM_LOG_WARNING("Skipping depth camera readback test because GPU device is unavailable.");
        runtime.shutdown();
        return 0;
    }

    GpuRenderTargetDesc targetDesc{};
    targetDesc.width = 640u;
    targetDesc.height = 480u;
    targetDesc.color = false;
    targetDesc.depth = true;
    targetDesc.debugName = "DepthCamera.Target";
    const GpuRenderTargetHandle target =
        graphicsDevice->renderTargetSystem().createRenderTarget(targetDesc);
    if (!graphicsDevice->renderTargetSystem().isValidRenderTarget(target))
    {
        runtime.shutdown();
        CRESSIM_LOG_ERROR("Failed to create depth camera target.");
        return 1;
    }

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.05f, -4.2f};
    world.setTransform(cameraEntity, cameraTransform);
    CameraComponent camera{};
    camera.product = CameraComponent::Product::Depth;
    camera.verticalFovDegrees = 52.0f;
    camera.output.mode = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
    camera.output.binding = cressim::neo::gpu::GpuRenderTargetBinding{target, 0u, 1u};
    camera.outputWidth = targetDesc.width;
    camera.outputHeight = targetDesc.height;
    camera.clearColor = false;
    camera.clearDepth = true;
    world.setCamera(cameraEntity, camera);

    auto &resources = runtime.getResources();
    const auto cubeMesh = resources.registerMesh(makeCubeMesh(0.65f));

    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "DepthCamera.Material";
    const auto material = resources.registerMaterial(materialDesc);

    const auto frontCubeEntity = world.createEntity();
    TransformComponent frontCubeTransform{};
    frontCubeTransform.worldTransform.position = {0.18f, -0.02f, -0.05f};
    frontCubeTransform.worldTransform.rotation = quaternionFromEulerDegrees(-18.0f, 32.0f, 0.0f);
    world.setTransform(frontCubeEntity, frontCubeTransform);
    MeshRendererComponent frontCube{};
    frontCube.mesh = cubeMesh;
    frontCube.material = material;
    frontCube.visible = false;
    world.setMeshRenderer(frontCubeEntity, frontCube);

    const auto backCubeEntity = world.createEntity();
    TransformComponent backCubeTransform{};
    backCubeTransform.worldTransform.position = {-0.14f, 0.03f, 1.35f};
    backCubeTransform.worldTransform.rotation = quaternionFromEulerDegrees(12.0f, -24.0f, 0.0f);
    backCubeTransform.worldTransform.scale = {1.35f, 1.35f, 1.35f};
    world.setTransform(backCubeEntity, backCubeTransform);
    MeshRendererComponent backCube{};
    backCube.mesh = cubeMesh;
    backCube.material = material;
    backCube.visible = true;
    world.setMeshRenderer(backCubeEntity, backCube);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    GpuRenderTargetDepthReadbackRequest firstRequest{};
    GpuRenderTargetDepthReadbackRequest secondRequest{};

    for (std::uint64_t frameIndex = 0u; frameIndex < 2u; ++frameIndex)
    {
        if (frameIndex == 1u)
        {
            frontCube.visible = true;
            world.setMeshRenderer(frontCubeEntity, frontCube);
        }

        runtime.prepare();
        const GpuRenderTargetDepthReadbackRequest request =
            graphicsDevice->renderTargetSystem().requestRenderTargetDepthReadback(
                cressim::neo::gpu::GpuRenderTargetBinding{target, 0u, 1u});
        if (request.id == 0u)
        {
            CRESSIM_LOG_ERROR("Failed to queue depth readback request.");
            runtime.shutdown();
            return 1;
        }
        if (frameIndex == 0u)
        {
            firstRequest = request;
        }
        else
        {
            secondRequest = request;
        }

        frame.frameIndex = frameIndex;
        frame.timeSeconds = static_cast<double>(frameIndex) * static_cast<double>(frame.deltaSeconds);
        (void)runtime.stepPhysics(frame);
        runtime.stepVisualSensors(frame);
        runtime.endFrame(frame);
    }

    GpuRenderTargetDepthReadbackEvent firstEvent{};
    GpuRenderTargetDepthReadbackEvent secondEvent{};
    const bool firstValid =
        graphicsDevice->renderTargetSystem().tryGetRenderTargetDepthReadback(firstRequest, firstEvent) &&
        cressim::neo::tests::helpers::isValidDepthReadback(firstEvent);
    const bool secondValid =
        graphicsDevice->renderTargetSystem().tryGetRenderTargetDepthReadback(secondRequest, secondEvent) &&
        cressim::neo::tests::helpers::isValidDepthReadback(secondEvent);
    runtime.shutdown();

    if (!firstValid || !secondValid)
    {
        CRESSIM_LOG_ERROR("Expected valid depth readback payloads for both frames.");
        return 1;
    }

    const float firstCenterDepth = cressim::neo::tests::helpers::decodeDepthValue(
        firstEvent, firstEvent.width / 2u, firstEvent.height / 2u);
    const float secondCenterDepth = cressim::neo::tests::helpers::decodeDepthValue(
        secondEvent, secondEvent.width / 2u, secondEvent.height / 2u);

    if (!(secondCenterDepth < firstCenterDepth))
    {
        CRESSIM_LOG_ERROR("Expected nearer front cube to reduce center depth. first=", firstCenterDepth,
                          " second=", secondCenterDepth);
        return 1;
    }

    CRESSIM_LOG_INFO("Depth camera readback passed. first center depth=", firstCenterDepth,
                     " second center depth=", secondCenterDepth);
    return 0;
}
