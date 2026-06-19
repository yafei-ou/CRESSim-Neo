#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "helpers/readback.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::ColliderComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::RigidBodyComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuDevice;
using cressim::neo::gpu::GpuRenderTargetDesc;
using cressim::neo::gpu::GpuRenderTargetHandle;
using cressim::neo::gpu::GpuRenderTargetReadbackEvent;
using cressim::neo::gpu::GpuRenderTargetReadbackRequest;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::tests::helpers::ReadbackPixel;

bool isNear(const ReadbackPixel &pixel, const ReadbackPixel &expected, const float tolerance)
{
    return std::fabs(pixel.r - expected.r) <= tolerance &&
           std::fabs(pixel.g - expected.g) <= tolerance &&
           std::fabs(pixel.b - expected.b) <= tolerance &&
           std::fabs(pixel.a - expected.a) <= tolerance;
}

bool containsNonClearPixel(const GpuRenderTargetReadbackEvent &event, const ReadbackPixel &clear)
{
    if (!cressim::neo::tests::helpers::isValidReadback(event))
    {
        return false;
    }

    for (std::uint32_t y = 0u; y < event.height; ++y)
    {
        for (std::uint32_t x = 0u; x < event.width; ++x)
        {
            if (!isNear(cressim::neo::tests::helpers::decodePixel(event, x, y), clear, 0.02f))
            {
                return true;
            }
        }
    }

    return false;
}

} // namespace

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping runtime staging test because runtime initialization failed.");
        return 0;
    }

    auto &world = runtime.getWorld();
    GpuDevice *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        CRESSIM_LOG_WARNING("Skipping runtime staging test because GPU device is unavailable.");
        runtime.shutdown();
        return 0;
    }

    GpuRenderTargetDesc targetDesc{};
    targetDesc.width            = 320u;
    targetDesc.height           = 240u;
    targetDesc.arraySize        = 1u;
    targetDesc.color            = true;
    targetDesc.depth            = true;
    targetDesc.layeredRendering = false;
    targetDesc.shaderReadable   = true;
    targetDesc.debugName        = "RuntimeStaging.Target";
    const GpuRenderTargetHandle target =
        device->renderTargetSystem().createRenderTarget(targetDesc);
    if (!device->renderTargetSystem().isValidRenderTarget(target))
    {
        CRESSIM_LOG_WARNING("Skipping runtime staging test because explicit target creation failed.");
        runtime.shutdown();
        return 0;
    }

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};
    CameraComponent camera{};
    camera.output.mode = cressim::neo::gpu::RenderOutputMode::ExplicitSurface;
    camera.output.binding = cressim::neo::gpu::GpuRenderTargetBinding{target, 0u, 1u};
    camera.outputWidth = targetDesc.width;
    camera.outputHeight = targetDesc.height;
    camera.clearColorValue = {0.0f, 0.0f, 0.0f, 1.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, camera);

    const auto lightEntity = world.createEntity();
    world.setDirectionalLight(lightEntity, DirectionalLightComponent{});

    auto &resources = runtime.getResources();
    MeshResourceDesc meshDesc{};
    meshDesc.debugName = "RuntimeStaging.Mesh";
    meshDesc.vertices = {
        {{-0.6f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.6f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 0.6f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f},
    };
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "RuntimeStaging.Material";
    materialDesc.baseColor = {0.95f, 0.25f, 0.20f};
    materialDesc.metallic = 0.1f;
    materialDesc.roughness = 0.35f;
    const auto material = resources.registerMaterial(materialDesc);

    const auto renderableEntity = world.createEntity();
    world.setTransform(renderableEntity, TransformComponent{});
    world.setMeshRenderer(renderableEntity, MeshRendererComponent{mesh, material, true});

    runtime.prepare();

    const GpuRenderTargetReadbackRequest firstRequest =
        device->renderTargetSystem().requestRenderTargetReadback(
            cressim::neo::gpu::GpuRenderTargetBinding{target, 0u, 1u});
    if (firstRequest.id == 0u)
    {
        CRESSIM_LOG_ERROR("Failed to queue frame-0 readback request.");
        runtime.shutdown();
        return 1;
    }

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    runtime.stepVisualSensors(frame);

    GpuRenderTargetReadbackEvent firstEvent{};
    if (!device->renderTargetSystem().tryGetRenderTargetReadback(firstRequest, firstEvent) ||
        !cressim::neo::tests::helpers::isValidReadback(firstEvent) ||
        !containsNonClearPixel(firstEvent, ReadbackPixel{0.0f, 0.0f, 0.0f, 1.0f}))
    {
        CRESSIM_LOG_ERROR("Expected authored frame-0 render to produce a visible readback.");
        runtime.shutdown();
        return 1;
    }

    TransformComponent hiddenTransform{};
    hiddenTransform.worldTransform.position = {100.0f, 0.0f, 0.0f};
    world.setTransform(renderableEntity, hiddenTransform);
    runtime.prepare();

    frame.frameIndex = 1u;
    frame.timeSeconds = static_cast<double>(frame.deltaSeconds);
    const GpuRenderTargetReadbackRequest secondRequest =
        device->renderTargetSystem().requestRenderTargetReadback(
            cressim::neo::gpu::GpuRenderTargetBinding{target, 0u, 1u});
    runtime.stepVisualSensors(frame);

    GpuRenderTargetReadbackEvent secondEvent{};
    if (!device->renderTargetSystem().tryGetRenderTargetReadback(secondRequest, secondEvent) ||
        !cressim::neo::tests::helpers::isValidReadback(secondEvent) ||
        containsNonClearPixel(secondEvent, ReadbackPixel{0.0f, 0.0f, 0.0f, 1.0f}))
    {
        CRESSIM_LOG_ERROR("Expected render() to use explicitly prepared authored transforms.");
        runtime.shutdown();
        return 1;
    }

    const auto rigidEntity = world.createEntity();
    TransformComponent rigidTransform{};
    rigidTransform.worldTransform.position = {0.0f, 0.0f, 0.0f};
    world.setTransform(rigidEntity, rigidTransform);

    RigidBodyComponent rigidBody{};
    rigidBody.linearVelocity = {1.5f, 0.0f, 0.0f};
    rigidBody.inverseMass = 1.0f;
    rigidBody.simulated = true;
    world.setRigidBody(rigidEntity, rigidBody);

    ColliderComponent rigidCollider{};
    rigidCollider.shapeType = cressim::neo::physics::ColliderShapeType::Sphere;
    rigidCollider.shapeParams = {0.5f, 0.0f, 0.0f, 0.0f};
    world.addCollider(rigidEntity, rigidCollider);
    runtime.prepare();

    constexpr std::uint32_t kPhysicsFrames = 10u;
    for (std::uint32_t i = 0u; i < kPhysicsFrames; ++i)
    {
        frame.frameIndex = 2u + i;
        frame.timeSeconds = static_cast<double>(frame.frameIndex) * frame.deltaSeconds;
        if (!runtime.stepPhysics(frame))
        {
            CRESSIM_LOG_ERROR("stepPhysics failed during staging test.");
            runtime.shutdown();
            return 1;
        }
    }

    const auto *finalRigidBody = world.physicsWorld().tryGetRigidBody(rigidEntity);
    if (finalRigidBody == nullptr)
    {
        CRESSIM_LOG_ERROR("Rigid body missing after staged physics steps.");
        runtime.shutdown();
        return 1;
    }

    const float expectedX =
        rigidBody.linearVelocity.x * frame.deltaSeconds * static_cast<float>(kPhysicsFrames);
    if (std::fabs(finalRigidBody->position.x - expectedX) > 0.05f)
    {
        CRESSIM_LOG_ERROR("Staged physics integration mismatch. expected=", expectedX,
                          " actual=", finalRigidBody->position.x);
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO("Runtime staging checks passed.");
    return 0;
}
