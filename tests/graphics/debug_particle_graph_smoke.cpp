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
#include <array>
#include <memory>
#include <vector>

namespace
{

std::uint32_t buildUniqueQueueFamilyIndices(const Diligent::IDeviceContext *firstContext,
                                            const Diligent::IDeviceContext *secondContext,
                                            std::array<std::uint32_t, 2> &outQueueFamilyIndices)
{
    std::uint32_t count      = 0u;
    const auto appendQueueId = [&outQueueFamilyIndices, &count](const Diligent::IDeviceContext *ctx)
    {
        if (ctx == nullptr)
        {
            return;
        }

        const std::uint32_t queueId = ctx->GetDesc().QueueId;
        for (std::uint32_t i = 0u; i < count; ++i)
        {
            if (outQueueFamilyIndices[i] == queueId)
            {
                return;
            }
        }

        if (count < outQueueFamilyIndices.size())
        {
            outQueueFamilyIndices[count++] = queueId;
        }
    };

    appendQueueId(firstContext);
    appendQueueId(secondContext);
    return count;
}

bool uploadRenderScene(cressim::neo::engine::RenderSceneUploader &uploader,
                       cressim::neo::engine::World &world,
                       const cressim::neo::graphics::RenderResourceManager &resources)
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
        !uploader.uploadLocalLightSelections(world.localLightSelections()))
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
    const auto &softEdges             = physicsWorld.softEdges();
    const auto &softBends             = physicsWorld.bendConstraints();
    const auto &softTets              = physicsWorld.softTets();
    const auto &softRenderData        = physicsWorld.softRenderData();
    const auto &curveRenderData       = physicsWorld.curveRenderData();
    const auto &rigidJoints           = physicsWorld.rigidJointScene();
    const auto &strandSegments        = physicsWorld.strandSegments();
    const auto &strandJoints          = physicsWorld.strandJoints();
    const auto &strandDistanceConstraints = physicsWorld.strandDistanceConstraints();
    const auto &rigidParticleAttachments   = physicsWorld.rigidParticleAttachments();
    const auto &strandRigidAttachments     = physicsWorld.strandRigidAttachments();
    const auto &rigidDistanceConstraints   = physicsWorld.rigidDistanceConstraints();
    const auto &routedCableConstraints     = physicsWorld.routedCableConstraints();
    const auto &routedCableRoutePoints     = physicsWorld.routedCableRoutePoints();
    const std::uint32_t bodyCount     = physicsWorld.rigidBodyCount();
    const std::uint32_t colliderCount = physicsWorld.colliderCount();
    std::uint32_t routedCableDebugSegmentCount = 0u;
    for (const auto &constraint : routedCableConstraints)
    {
        if (constraint.routePointCount > 1u)
        {
            routedCableDebugSegmentCount += constraint.routePointCount - 1u;
        }
    }
    const std::uint32_t curveRenderVertexCount = [&curveRenderData]()
    {
        std::uint32_t totalVertexCount = 0u;
        for (const auto &descriptor : curveRenderData.descriptors)
        {
            totalVertexCount =
                std::max(totalVertexCount, descriptor.vertexBase + descriptor.vertexCount);
        }
        return totalVertexCount;
    }();

    const Diligent::Uint64 sharedContextMask =
        cressim::neo::gpu::contextMaskForId(computeBackend.contextId) |
        cressim::neo::gpu::contextMaskForId(graphicsBackend.contextId);
    std::array<std::uint32_t, 2> sharedQueueFamilyIndices{};
    const std::uint32_t sharedQueueFamilyIndexCount = buildUniqueQueueFamilyIndices(
        computeBackend.computeContext, graphicsBackend.graphicsContext, sharedQueueFamilyIndices);

    if (!sceneState.ensureCapacity(
            computeBackend.renderDevice, bodyCount, colliderCount,
            static_cast<std::uint32_t>(particles.size()), physicsWorld.fluidCount(),
            static_cast<std::uint32_t>(physicsWorld.particleContactMaterials().size()),
            static_cast<std::uint32_t>(physicsWorld.fluidMaterials().size()),
            static_cast<std::uint32_t>(softEdges.size()),
            static_cast<std::uint32_t>(softBends.size()),
            static_cast<std::uint32_t>(softTets.size()),
            static_cast<std::uint32_t>(strandSegments.size()),
            static_cast<std::uint32_t>(strandJoints.size()),
            static_cast<std::uint32_t>(strandDistanceConstraints.size()),
            static_cast<std::uint32_t>(rigidJoints.ball.size()),
            static_cast<std::uint32_t>(rigidJoints.spherical.size()),
            static_cast<std::uint32_t>(rigidJoints.hinge.size()),
            static_cast<std::uint32_t>(rigidJoints.slider.size()),
            static_cast<std::uint32_t>(rigidParticleAttachments.size()),
            static_cast<std::uint32_t>(strandRigidAttachments.size()),
            static_cast<std::uint32_t>(rigidDistanceConstraints.size()),
            static_cast<std::uint32_t>(softRenderData.fallbackNormals.size()),
            static_cast<std::uint32_t>(softRenderData.vertexTriangleIndices.size()),
            static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size()),
            std::max<std::uint32_t>(
                static_cast<std::uint32_t>(softRenderData.softBodyParticleRanges.size()), 1u),
            std::max<std::uint32_t>(physicsWorld.softBodyBoundsChunkCount(), 1u),
            static_cast<std::uint32_t>(physicsWorld.suturingPairs().size()),
            physicsWorld.reservedSuturingPathHeaderCount(),
            physicsWorld.reservedSuturingPathNodeCount(),
            static_cast<std::uint32_t>(routedCableConstraints.size()),
            static_cast<std::uint32_t>(routedCableRoutePoints.size()),
            routedCableDebugSegmentCount,
            static_cast<std::uint32_t>(curveRenderData.descriptors.size()),
            static_cast<std::uint32_t>(curveRenderData.particleIndices.size()),
            curveRenderVertexCount,
            sharedContextMask, sharedQueueFamilyIndices.data(), sharedQueueFamilyIndexCount,
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
    graphics::MeshResourceDesc softSurfaceMesh{};
    softSurfaceMesh.debugName = "DebugParticleGraphSmoke.SoftSurface";
    softSurfaceMesh.vertices.resize(3u);
    softSurfaceMesh.vertices[0].position = {-0.5f, 0.0f, 0.0f};
    softSurfaceMesh.vertices[1].position = {0.0f, 0.0f, 0.0f};
    softSurfaceMesh.vertices[2].position = {0.0f, 0.5f, 0.0f};
    softSurfaceMesh.indices              = {0u, 1u, 2u};
    graphics::MaterialResourceDesc softSurfaceMaterial{};
    softSurfaceMaterial.debugName = "DebugParticleGraphSmoke.SoftSurfaceMaterial";
    world.setMeshRenderer(
        softEntity,
        engine::MeshRendererComponent{resources.registerMesh(softSurfaceMesh),
                                      resources.registerMaterial(softSurfaceMaterial), true});

    physics::PhysicsSceneGpuState physicsSceneState;
    if (!uploadRenderScene(uploader, world, resources) ||
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
        metadataOverride.front().deformableIndex = 0u;
        metadataOverride.front().deformableType =
            static_cast<std::uint32_t>(graphics::GpuRenderableDeformableType::SoftBody);
        hostSceneView.renderableMetadata = &metadataOverride;
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
