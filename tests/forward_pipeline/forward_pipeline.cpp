#include "common/frame_context.h"
#include "engine/components.h"
#include "engine/runtime.h"

#include <iostream>

namespace
{

using cressim::neo::common::FrameContext;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::engine::TransformComponent;
using cressim::neo::graphics::BlendMode;
using cressim::neo::graphics::GraphicsBackend;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::RenderStats;

bool sameStats(const RenderStats& lhs, const RenderStats& rhs)
{
    return lhs.drawCalls == rhs.drawCalls &&
        lhs.opaqueDrawCalls == rhs.opaqueDrawCalls &&
        lhs.shadowDrawCalls == rhs.shadowDrawCalls &&
        lhs.transparentDrawCalls == rhs.transparentDrawCalls &&
        lhs.renderableCount == rhs.renderableCount &&
        lhs.validRenderableCount == rhs.validRenderableCount &&
        lhs.culledRenderableCount == rhs.culledRenderableCount &&
        lhs.opaqueQueueCount == rhs.opaqueQueueCount &&
        lhs.shadowCasterQueueCount == rhs.shadowCasterQueueCount &&
        lhs.transparentQueueCount == rhs.transparentQueueCount &&
        lhs.lightCount == rhs.lightCount &&
        lhs.cameraCount == rhs.cameraCount;
}

} // namespace

int main()
{
    RuntimeConfig config{};
    config.graphicsDeviceDesc.preferredBackend = GraphicsBackend::Null;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        std::cerr << "Runtime initialization failed.\n";
        return 1;
    }

    auto& world = runtime.getWorld();

    const auto cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 0.0f, 0.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    const auto lightEntity = world.createEntity();
    world.setDirectionalLight(lightEntity, DirectionalLightComponent{});

    auto& resources = runtime.getResources();

    MeshResourceDesc meshDesc{};
    meshDesc.debugName = "ForwardPipeline.TestTriangle";
    meshDesc.vertices = {
        {{-0.4f, -0.4f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.4f, -0.4f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 0.4f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f}};
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    MaterialResourceDesc opaqueMaterialDesc{};
    opaqueMaterialDesc.debugName = "ForwardPipeline.Opaque";
    const auto opaqueMaterial = resources.registerMaterial(opaqueMaterialDesc);

    MaterialResourceDesc transparentMaterialDesc{};
    transparentMaterialDesc.debugName = "ForwardPipeline.Transparent";
    transparentMaterialDesc.blendMode = BlendMode::Transparent;
    transparentMaterialDesc.opacity = 0.5f;
    transparentMaterialDesc.castsShadows = false;
    const auto transparentMaterial = resources.registerMaterial(transparentMaterialDesc);

    const auto visibleOpaqueEntity = world.createEntity();
    TransformComponent visibleOpaqueTransform{};
    visibleOpaqueTransform.worldTransform.position = {0.0f, 0.0f, 2.0f};
    world.setTransform(visibleOpaqueEntity, visibleOpaqueTransform);
    MeshRendererComponent visibleOpaqueRenderer{};
    visibleOpaqueRenderer.mesh = mesh;
    visibleOpaqueRenderer.material = opaqueMaterial;
    visibleOpaqueRenderer.visible = true;
    world.setMeshRenderer(visibleOpaqueEntity, visibleOpaqueRenderer);

    const auto visibleTransparentEntity = world.createEntity();
    TransformComponent visibleTransparentTransform{};
    visibleTransparentTransform.worldTransform.position = {0.8f, 0.0f, 3.0f};
    world.setTransform(visibleTransparentEntity, visibleTransparentTransform);
    MeshRendererComponent visibleTransparentRenderer{};
    visibleTransparentRenderer.mesh = mesh;
    visibleTransparentRenderer.material = transparentMaterial;
    visibleTransparentRenderer.visible = true;
    world.setMeshRenderer(visibleTransparentEntity, visibleTransparentRenderer);

    const auto culledEntity = world.createEntity();
    TransformComponent culledTransform{};
    culledTransform.worldTransform.position = {0.0f, 0.0f, -2.0f};
    world.setTransform(culledEntity, culledTransform);
    MeshRendererComponent culledRenderer{};
    culledRenderer.mesh = mesh;
    culledRenderer.material = opaqueMaterial;
    culledRenderer.visible = true;
    world.setMeshRenderer(culledEntity, culledRenderer);

    FrameContext frame{};
    frame.deltaSeconds = 1.0f / 60.0f;
    frame.frameIndex = 0;
    frame.timeSeconds = 0.0;
    runtime.tick(frame);
    const RenderStats firstFrame = runtime.lastRenderStats();

    frame.frameIndex = 1;
    frame.timeSeconds = frame.deltaSeconds;
    runtime.tick(frame);
    const RenderStats secondFrame = runtime.lastRenderStats();

    runtime.shutdown();

    if (firstFrame.renderableCount != 3 || firstFrame.validRenderableCount != 3)
    {
        std::cerr << "Unexpected renderable counters.\n";
        return 1;
    }
    if (firstFrame.culledRenderableCount != 1)
    {
        std::cerr << "Unexpected culling count. expected=1 got=" << firstFrame.culledRenderableCount << '\n';
        return 1;
    }
    if (firstFrame.opaqueQueueCount != 1 || firstFrame.transparentQueueCount != 1 || firstFrame.shadowCasterQueueCount != 1)
    {
        std::cerr << "Unexpected queue counters. opaque=" << firstFrame.opaqueQueueCount
                  << " transparent=" << firstFrame.transparentQueueCount
                  << " shadow=" << firstFrame.shadowCasterQueueCount << '\n';
        return 1;
    }
    if (firstFrame.transparentDrawCalls != 0 || firstFrame.shadowDrawCalls != 0)
    {
        std::cerr << "Transparent/shadow pass should be scaffold-only in milestone 1.\n";
        return 1;
    }
    if (firstFrame.cameraCount != 1 || firstFrame.lightCount != 1)
    {
        std::cerr << "Unexpected camera/light counters.\n";
        return 1;
    }
    if (!sameStats(firstFrame, secondFrame))
    {
        std::cerr << "Forward queue statistics were not stable across identical frames.\n";
        return 1;
    }

    std::cout << "Forward pipeline null-backend checks passed.\n";
    return 0;
}
