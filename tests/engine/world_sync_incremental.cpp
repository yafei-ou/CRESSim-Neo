#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"
#include "common/logger.h"


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

void tick(Runtime& runtime, FrameContext& frame)
{
    runtime.prepare();
    const bool physicsStepSucceeded = runtime.uploadWorld() && runtime.stepPhysics(frame);
    if (physicsStepSucceeded)
    {
        (void)runtime.stepSimulationSensors(frame);
    }
    runtime.stepVisualSensors(frame);
    runtime.endFrame(frame);
    frame.frameIndex += 1u;
    frame.timeSeconds += frame.deltaSeconds;
}

} // namespace

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR( "Runtime initialization failed.\n");
        return 1;
    }

    auto& world = runtime.getWorld();
    auto& resources = runtime.getResources();

    MeshResourceDesc meshDesc{};
    meshDesc.debugName = "WorldSync.Mesh";
    meshDesc.vertices = {
        {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f}};
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "WorldSync.Material";
    const auto material = resources.registerMaterial(materialDesc);

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    world.setDirectionalLight(lightEntity, DirectionalLightComponent{});

    const auto renderableEntity = world.createEntity();
    world.setTransform(renderableEntity, TransformComponent{});
    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(renderableEntity, renderer);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;

    tick(runtime, frame);
    const auto stats0 = runtime.lastRenderStats();
    if (stats0.renderableCount != 1)
    {
        CRESSIM_LOG_ERROR( "Unexpected stats after initial frame.\n");
        runtime.shutdown();
        return 1;
    }

    tick(runtime, frame);
    const auto stats1 = runtime.lastRenderStats();
    if (stats1.renderableCount != 1)
    {
        CRESSIM_LOG_ERROR( "Expected stable render count on unchanged frame.\n");
        runtime.shutdown();
        return 1;
    }

    TransformComponent movedTransform{};
    movedTransform.worldTransform.position = {1.0f, 0.0f, 0.0f};
    world.setTransform(renderableEntity, movedTransform);
    tick(runtime, frame);
    const auto stats2 = runtime.lastRenderStats();
    if (stats2.renderableCount != 1)
    {
        CRESSIM_LOG_ERROR( "Expected sync execution after transform update.\n");
        runtime.shutdown();
        return 1;
    }

    if (!world.removeMeshRenderer(renderableEntity))
    {
        CRESSIM_LOG_ERROR( "Failed to remove mesh renderer.\n");
        runtime.shutdown();
        return 1;
    }
    tick(runtime, frame);
    const auto stats3 = runtime.lastRenderStats();
    if (stats3.renderableCount != 0)
    {
        CRESSIM_LOG_ERROR( "Expected renderable removal to be reflected in world-owned host scene.\n");
        runtime.shutdown();
        return 1;
    }

    tick(runtime, frame);
    const auto stats4 = runtime.lastRenderStats();
    if (stats4.renderableCount != 0)
    {
        CRESSIM_LOG_ERROR( "Expected stable post-removal render count.\n");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();
    CRESSIM_LOG_INFO( "Incremental world sync checks passed.\n");
    return 0;
}
