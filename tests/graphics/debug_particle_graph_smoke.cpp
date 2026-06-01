#include "common/frame_context.h"
#include "common/logger.h"
#include "engine/components.h"
#include "engine/render_scene_uploader.h"
#include "engine/world.h"
#include "gpu/gpu_device.h"
#include "graphics/renderer.h"
#include "graphics/render_resource_manager.h"
#include "physics/physics_scene_gpu_state.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace
{

bool uploadRenderScene(cressim::neo::engine::RenderSceneUploader &uploader,
                       cressim::neo::engine::World &world,
                       const cressim::neo::graphics::RenderResourceManager &resources,
                       cressim::neo::gpu::GpuDevice &device)
{
    world.ensureRenderStateUpToDate(resources);
    if (!uploader.uploadEntityPoseData(world.renderObjectPositions(),
                                       world.renderObjectOrientations(),
                                       world.renderObjectScales()) ||
        !uploader.uploadRenderableMetadata(world.renderableMetadata()) ||
        !uploader.uploadRenderableQueueInfo(world.renderableQueueInfo()) ||
        !uploader.uploadSoftBodyVertexBindings(world.softBodyVertexBindings()) ||
        !uploader.uploadCameraInputs(world.cameraInputs()) ||
        !uploader.uploadLightInputs(world.lightInputs()) ||
        !uploader.uploadLocalLightSelections(world.localLightSelections()) ||
        !device.waitForPhysicsOnGraphics())
    {
        return false;
    }

    world.setGpuEntityScene(uploader.sceneView());
    return true;
}

bool uploadPhysicsScene(cressim::neo::physics::PhysicsSceneGpuState &sceneState,
                        cressim::neo::engine::World &world,
                        cressim::neo::gpu::GpuDevice &device)
{
    cressim::neo::gpu::GpuComputeBackendContext computeBackend{};
    cressim::neo::gpu::GpuGraphicsBackendContext graphicsBackend{};
    if (!device.tryGetPhysicsBackendContext(computeBackend) ||
        !device.tryGetGraphicsBackendContext(graphicsBackend) ||
        computeBackend.renderDevice == nullptr || computeBackend.computeContext == nullptr ||
        graphicsBackend.renderDevice == nullptr)
    {
        return false;
    }

    auto &physicsWorld = world.physicsWorld();
    physicsWorld.ensureDerivedStateUpToDate();
    const auto &particles          = physicsWorld.particles();
    const auto &softEdges          = physicsWorld.softEdges();
    const auto &softTets           = physicsWorld.softTets();
    const auto &softRenderData     = physicsWorld.softRenderData();
    const auto &rigidJoints        = physicsWorld.rigidJointScene();
    const std::uint32_t bodyCount  = physicsWorld.rigidBodyCount();
    const std::uint32_t colliderCount = physicsWorld.colliderCount();

    const Diligent::Uint64 sharedContextMask =
        cressim::neo::gpu::contextMaskForId(computeBackend.contextId) |
        cressim::neo::gpu::contextMaskForId(graphicsBackend.contextId);

    if (!sceneState.ensureCapacity(
            computeBackend.renderDevice, bodyCount, colliderCount,
            static_cast<std::uint32_t>(particles.size()), physicsWorld.fluidCount(),
            static_cast<std::uint32_t>(physicsWorld.particleContactMaterials().size()),
            static_cast<std::uint32_t>(physicsWorld.fluidMaterials().size()),
            static_cast<std::uint32_t>(softEdges.size()),
            static_cast<std::uint32_t>(softTets.size()),
            static_cast<std::uint32_t>(rigidJoints.ball.size()),
            static_cast<std::uint32_t>(rigidJoints.hinge.size()),
            static_cast<std::uint32_t>(rigidJoints.slider.size()),
            static_cast<std::uint32_t>(softRenderData.fallbackNormals.size()),
            static_cast<std::uint32_t>(softRenderData.vertexTriangleIndices.size()),
            static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size()),
            std::max<std::uint32_t>(
                static_cast<std::uint32_t>(softRenderData.softBodyParticleRanges.size()), 1u),
            std::max<std::uint32_t>(physicsWorld.softBodyBoundsChunkCount(), 1u),
            sharedContextMask,
            device.supportsNativePhysicsFloatAtomics()))
    {
        return false;
    }

    return sceneState.uploadWorldState(computeBackend.computeContext, physicsWorld, bodyCount,
                                       colliderCount);
}

} // namespace

int main()
{
    using namespace cressim::neo;

    gpu::GpuDeviceDesc deviceDesc{};
    deviceDesc.preferredBackend = gpu::GpuBackend::Vulkan;
    deviceDesc.enableValidation = false;

    std::unique_ptr<gpu::GpuDevice> device = gpu::createGpuDevice();
    if (!device || !device->initialize(deviceDesc))
    {
        CRESSIM_LOG_WARNING(
            "Skipping debug particle graph smoke because GPU initialization failed.\n");
        return 0;
    }

    graphics::RenderResourceManager resources;
    engine::RenderSceneUploader uploader(*device);
    if (!uploader.initialize())
    {
        CRESSIM_LOG_WARNING(
            "Skipping debug particle graph smoke because scene uploader initialization failed.\n");
        device->shutdown();
        return 0;
    }

    auto renderer = std::make_unique<graphics::Renderer>(*device, resources);
    if (!renderer->initialize())
    {
        CRESSIM_LOG_WARNING(
            "Skipping debug particle graph smoke because renderer initialization failed.\n");
        renderer.reset();
        uploader.shutdown();
        device->shutdown();
        return 0;
    }

    engine::World world;

    const common::EntityId cameraEntity = world.createEntity();
    engine::TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.5f, -4.0f};
    world.setTransform(cameraEntity, cameraTransform);

    engine::CameraComponent camera{};
    camera.outputWidth = 640u;
    camera.outputHeight = 360u;
    world.setCamera(cameraEntity, camera);

    const common::EntityId softEntity = world.createEntity();
    engine::MeshfreeSoftBodyComponent softBody{};
    softBody.particles = {
        {-0.5f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.5f, 0.0f, 0.0f},
        {0.0f, 0.5f, 0.0f},
    };
    softBody.staticParticleIndices = {0u};
    softBody.neighbourCount = 2u;
    softBody.particleRadius = 0.04f;
    if (!world.setMeshfreeSoftBody(softEntity, softBody))
    {
        CRESSIM_LOG_ERROR("Failed to author debug meshfree soft body.\n");
        renderer.reset();
        uploader.shutdown();
        device->shutdown();
        return 1;
    }

    physics::PhysicsSceneGpuState physicsSceneState;
    if (!uploadRenderScene(uploader, world, resources, *device) ||
        !uploadPhysicsScene(physicsSceneState, world, *device) ||
        !device->waitForPhysicsOnGraphics())
    {
        CRESSIM_LOG_ERROR("Failed to upload debug particle graph smoke scene.\n");
        renderer.reset();
        uploader.shutdown();
        device->shutdown();
        return 1;
    }

    graphics::RenderFrameOptions options{};
    options.debugParticles.enabled = true;
    options.debugParticles.drawConstraintEdges = true;
    options.debugParticles.highlightStaticParticles = true;

    common::FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    const physics::PhysicsGpuSceneView physicsSceneView = physicsSceneState.sceneView();
    graphics::HostSceneView hostSceneView = world.hostSceneView();
    std::vector<graphics::GpuRenderableMetadata> metadataOverride = world.renderableMetadata();
    if (!metadataOverride.empty())
    {
        metadataOverride.front().flags =
            static_cast<std::uint32_t>(graphics::GpuRenderableFlags::Active);
        metadataOverride.front().softBodyIndex = 0u;
        hostSceneView.renderableMetadata       = &metadataOverride;
    }
    const graphics::RenderStats stats =
        renderer->render(frame, hostSceneView, &physicsSceneView, options);
    if (stats.renderedCameraCount == 0u)
    {
        CRESSIM_LOG_ERROR("Debug particle graph smoke rendered no cameras.\n");
        renderer.reset();
        uploader.shutdown();
        device->shutdown();
        return 1;
    }

    renderer.reset();
    uploader.shutdown();
    device->shutdown();
    CRESSIM_LOG_INFO("Debug particle graph smoke checks passed.\n");
    return 0;
}
