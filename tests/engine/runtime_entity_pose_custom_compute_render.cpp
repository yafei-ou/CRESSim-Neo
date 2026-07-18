#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/custom_compute.h"
#include "engine/runtime.h"
#include "../helpers/readback.h"

#include <cmath>
#include <cstdint>

namespace
{

constexpr const char *kCameraOrientationShader = R"(
#include "structured_buffer_compat.hlsli"

cbuffer CameraOrientationConstants
{
    uint poseSlot;
    uint padding0;
    uint padding1;
    uint padding2;
    float4 orientation;
};

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityOrientations);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CRESSIM_SB_STORE(g_EntityOrientations, poseSlot, orientation);
}
)";

constexpr std::uint32_t kTargetSegmentationId = 77u;

struct CameraOrientationConstants
{
    std::uint32_t poseSlot = 0u;
    std::uint32_t padding0 = 0u;
    std::uint32_t padding1 = 0u;
    std::uint32_t padding2 = 0u;
    Diligent::float4 orientation{};
};

Diligent::float4 quaternionFromYawRadians(const float yawRadians)
{
    return {0.0f, std::sin(0.5f * yawRadians), 0.0f, std::cos(0.5f * yawRadians)};
}

cressim::neo::graphics::MeshResourceDesc makeCubeMesh(const float halfExtent,
                                                      const char *debugName)
{
    cressim::neo::graphics::MeshResourceDesc mesh{};
    mesh.debugName = debugName;
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
    using namespace cressim::neo;

    engine::RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    config.gpuDeviceDesc.enableValidation = false;
    config.sceneLayout.envCount           = 1u;

    engine::Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING("Skipping entity-pose custom-compute render test because runtime "
                            "initialization failed.");
        return 0;
    }

    auto *device = runtime.getGpuDevice();
    if (device == nullptr)
    {
        CRESSIM_LOG_WARNING("Skipping entity-pose custom-compute render test because GPU device "
                            "is unavailable.");
        runtime.shutdown();
        return 0;
    }

    auto &world = runtime.getWorld();
    auto &resources = runtime.getResources();

    gpu::GpuRenderTargetDesc targetDesc{};
    targetDesc.width            = 96u;
    targetDesc.height           = 96u;
    targetDesc.arraySize        = 1u;
    targetDesc.color            = true;
    targetDesc.depth            = true;
    targetDesc.layeredRendering = false;
    targetDesc.shaderReadable   = true;
    targetDesc.colorFormat      = Diligent::TEX_FORMAT_R32_UINT;
    targetDesc.debugName        = "RuntimeEntityPoseCustomComputeRender.Target";
    const gpu::GpuRenderTargetHandle target =
        device->renderTargetSystem().createRenderTarget(targetDesc);
    if (!device->renderTargetSystem().isValidRenderTarget(target))
    {
        CRESSIM_LOG_ERROR("Failed to create entity-pose render target.");
        runtime.shutdown();
        return 1;
    }

    const graphics::MeshHandle cubeMesh =
        resources.registerMesh(makeCubeMesh(0.35f, "RuntimeEntityPoseCustomComputeRender.Cube"));
    graphics::MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "RuntimeEntityPoseCustomComputeRender.Material";
    const graphics::MaterialHandle material = resources.registerMaterial(materialDesc);

    const common::EntityId cubeEntity = world.createEntity();
    engine::TransformComponent cubeTransform{};
    cubeTransform.worldTransform.position = {1.0f, 0.0f, 0.5f};
    world.setTransform(cubeEntity, cubeTransform);
    engine::MeshRendererComponent cubeRenderer{};
    cubeRenderer.mesh           = cubeMesh;
    cubeRenderer.material       = material;
    cubeRenderer.segmentationId = kTargetSegmentationId;
    cubeRenderer.visible        = true;
    world.setMeshRenderer(cubeEntity, cubeRenderer);

    const common::EntityId cameraEntity = world.createEntity();
    engine::TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -4.0f};
    world.setTransform(cameraEntity, cameraTransform);
    engine::CameraComponent camera{};
    camera.product        = engine::CameraComponent::Product::SegmentationDepth;
    camera.output.mode    = gpu::RenderOutputMode::ExplicitSurface;
    camera.output.binding = gpu::GpuRenderTargetBinding{target, 0u, 1u};
    camera.outputWidth    = targetDesc.width;
    camera.outputHeight   = targetDesc.height;
    world.setCamera(cameraEntity, camera);

    runtime.prepare();
    if (!runtime.uploadWorld())
    {
        CRESSIM_LOG_ERROR("Failed to upload world before entity-pose custom-compute render test.");
        runtime.shutdown();
        return 1;
    }

    const std::vector<engine::CustomComputeResourceDesc> resourcesExposed =
        runtime.listCustomComputeResources();
    const auto hasResource = [&](const char *key)
    {
        for (const auto &resource : resourcesExposed)
        {
            if (resource.key == key)
            {
                return true;
            }
        }
        return false;
    };
    if (!hasResource("entity.orientations"))
    {
        CRESSIM_LOG_ERROR("Expected entity.orientations custom-compute resource to be exposed.");
        runtime.shutdown();
        return 1;
    }

    const std::uint32_t cameraPoseSlot = world.entityPoseSlot(cameraEntity);
    if (cameraPoseSlot == 0xffffffffu)
    {
        CRESSIM_LOG_ERROR("Camera pose slot was invalid in entity-pose render test.");
        runtime.shutdown();
        return 1;
    }

    CameraOrientationConstants constants{};
    constants.poseSlot    = cameraPoseSlot;
    constants.orientation = quaternionFromYawRadians(std::atan2(1.0f, 4.5f));
    const auto *constantBytes = reinterpret_cast<const std::uint8_t *>(&constants);

    engine::CustomComputePassDesc passDesc{};
    passDesc.debugName        = "RuntimeEntityPoseCustomComputeRender.Pass";
    passDesc.shaderSource     = kCameraOrientationShader;
    passDesc.threadGroupSizeX = 1u;
    passDesc.resourceBindings.resize(1u);
    passDesc.resourceBindings[0].shaderVariableName = "g_EntityOrientations";
    passDesc.resourceBindings[0].resourceKey        = "entity.orientations";
    passDesc.resourceBindings[0].access = engine::CustomComputeResourceAccess::ReadWrite;
    passDesc.constantBufferVariableName = "CameraOrientationConstants";
    passDesc.constantBufferSizeBytes    = sizeof(CameraOrientationConstants);
    passDesc.constantData.assign(constantBytes, constantBytes + sizeof(CameraOrientationConstants));
    passDesc.dispatch.mode        = engine::CustomComputeDispatchMode::ExplicitGroupCount;
    passDesc.dispatch.groupCountX = 1u;

    const engine::CustomComputePassHandle pass = runtime.createCustomComputePass(passDesc);
    if (!pass.isValid())
    {
        CRESSIM_LOG_ERROR("Failed to create entity-pose orientation custom-compute pass.");
        runtime.shutdown();
        return 1;
    }

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    const gpu::GpuRenderTargetReadbackRequest initialReadback =
        device->renderTargetSystem().requestRenderTargetReadback(
            gpu::GpuRenderTargetBinding{target, 0u, 1u});
    if (initialReadback.id == 0u)
    {
        CRESSIM_LOG_ERROR("Failed to request initial render readback.");
        runtime.shutdown();
        return 1;
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

    gpu::GpuRenderTargetReadbackEvent initialEvent{};
    if (!device->renderTargetSystem().tryGetRenderTargetReadback(initialReadback, initialEvent) ||
        !tests::helpers::isValidReadback(initialEvent))
    {
        CRESSIM_LOG_WARNING("Skipping entity-pose render test because initial readback failed.");
        runtime.shutdown();
        return 0;
    }
    const std::uint32_t initialCenter =
        tests::helpers::decodeUint32Pixel(initialEvent, initialEvent.width / 2u,
                                          initialEvent.height / 2u);

    frame.frameIndex = 1u;
    frame.timeSeconds = frame.deltaSeconds;
    const gpu::GpuRenderTargetReadbackRequest updatedReadback =
        device->renderTargetSystem().requestRenderTargetReadback(
            gpu::GpuRenderTargetBinding{target, 0u, 1u});
    if (updatedReadback.id == 0u)
    {
        CRESSIM_LOG_ERROR("Failed to request updated render readback.");
        runtime.shutdown();
        return 1;
    }
    if (!runtime.executeCustomComputePass(pass))
    {
        CRESSIM_LOG_ERROR("Failed to execute entity-pose orientation custom-compute pass.");
        runtime.shutdown();
        return 1;
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);

    gpu::GpuRenderTargetReadbackEvent updatedEvent{};
    if (!device->renderTargetSystem().tryGetRenderTargetReadback(updatedReadback, updatedEvent) ||
        !tests::helpers::isValidReadback(updatedEvent))
    {
        CRESSIM_LOG_WARNING("Skipping entity-pose render test because updated readback failed.");
        runtime.shutdown();
        return 0;
    }
    const std::uint32_t updatedCenter =
        tests::helpers::decodeUint32Pixel(updatedEvent, updatedEvent.width / 2u,
                                          updatedEvent.height / 2u);

    if (initialCenter == kTargetSegmentationId || updatedCenter != kTargetSegmentationId)
    {
        CRESSIM_LOG_ERROR("Expected entity-pose custom compute to move the target into the image "
                          "center. initialCenter=", initialCenter, " updatedCenter=",
                          updatedCenter, ".");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    return 0;
}
