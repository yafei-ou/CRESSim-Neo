#include "common/logger.h"
#include "engine/components.h"
#include "engine/world.h"
#include "graphics/render_resource_manager.h"

namespace
{

using cressim::neo::engine::DirectionalLightComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::World;
using cressim::neo::graphics::GpuLightType;
using cressim::neo::common::SceneLayoutDesc;
using cressim::neo::graphics::kMainDirectionalLightSlot;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MaterialRenderMode;
using cressim::neo::graphics::MeshResourceDesc;
using cressim::neo::graphics::RenderResourceManager;

} // namespace

int main()
{
    World world;
    RenderResourceManager resources;

    SceneLayoutDesc layout{};
    layout.envCount = 1u;
    layout.maxLightsPerEnv = 4u;
    world.setSceneLayout(layout);

    MeshResourceDesc meshDesc{};
    meshDesc.debugName = "LocalShadowRegistry.Mesh";
    meshDesc.vertices = {
        {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f},
    };
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "LocalShadowRegistry.Material";
    const auto material = resources.registerMaterial(materialDesc);
    MaterialResourceDesc transparentMaterialDesc{};
    transparentMaterialDesc.debugName = "LocalShadowRegistry.TransparentMaterial";
    transparentMaterialDesc.renderMode = MaterialRenderMode::Transparent;
    transparentMaterialDesc.castsShadows = false;
    const auto transparentMaterial = resources.registerMaterial(transparentMaterialDesc);

    const auto renderableEntity = world.createEntity();
    TransformComponent renderableTransform{};
    renderableTransform.worldTransform.position = {2.0f, 0.5f, -1.0f};
    world.setTransform(renderableEntity, renderableTransform);
    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(renderableEntity, renderer);

    const auto transparentEntity = world.createEntity();
    TransformComponent transparentTransform{};
    transparentTransform.worldTransform.position = {-1.5f, 0.0f, 0.0f};
    world.setTransform(transparentEntity, transparentTransform);
    MeshRendererComponent transparentRenderer{};
    transparentRenderer.mesh = mesh;
    transparentRenderer.material = transparentMaterial;
    transparentRenderer.visible = true;
    world.setMeshRenderer(transparentEntity, transparentRenderer);

    const auto mainLightEntity = world.createEntity();
    DirectionalLightComponent mainLight{};
    mainLight.direction = {0.2f, -1.0f, 0.1f};
    mainLight.castsShadows = true;
    world.setDirectionalLight(mainLightEntity, mainLight);

    const auto localDirectionalEntity = world.createEntity();
    DirectionalLightComponent localDirectional{};
    localDirectional.direction = {-0.3f, -1.0f, 0.25f};
    localDirectional.castsShadows = true;
    world.setDirectionalLight(localDirectionalEntity, localDirectional);

    world.ensureRenderStateUpToDate(resources);

    const auto &lights = world.lights();
    if (lights.empty() || lights[0].lightSlot != kMainDirectionalLightSlot ||
        lights[0].type != GpuLightType::Directional)
    {
        CRESSIM_LOG_ERROR("Expected slot 0 to remain the main directional light.");
        return 1;
    }

    const auto &selection = world.localLightSelections().front();
    if (selection.localLightCount != 1u || selection.shadowedLocalLightCount != 1u ||
        selection.shadowedPointLightCount != 0u)
    {
        CRESSIM_LOG_ERROR("Expected exactly one non-main directional local shadow light.");
        return 1;
    }

    const std::uint32_t localLightIndex = selection.lightIndices[0];
    if (localLightIndex >= lights.size() || lights[localLightIndex].type != GpuLightType::Directional ||
        lights[localLightIndex].lightSlot == kMainDirectionalLightSlot)
    {
        CRESSIM_LOG_ERROR("Expected local-light selection to include the non-main directional light.");
        return 1;
    }

    const auto &shadowRegistry = world.shadowDrawRegistry();
    const auto &transparentRegistry = world.transparentDrawRegistry();
    const auto &localShadowRegistry = world.localShadowDrawRegistry();
    if (shadowRegistry.size() != 4u)
    {
        CRESSIM_LOG_ERROR("Expected main-light shadow registry to remain cascade-expanded.");
        return 1;
    }
    if (localShadowRegistry.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Expected local shadow registry to contain one non-cascade bucket.");
        return 1;
    }
    if (transparentRegistry.size() != 1u)
    {
        CRESSIM_LOG_ERROR("Expected transparent registry to contain one direct-draw entry.");
        return 1;
    }

    const auto &queueInfo = world.renderableQueueInfo();
    if (queueInfo.empty() || queueInfo[0].opaqueCommandIndex != 0u ||
        queueInfo[0].shadowCommandBaseIndex != 0u ||
        queueInfo[0].localShadowCommandIndex != 0u)
    {
        CRESSIM_LOG_ERROR("Expected renderable queue info to expose opaque, main-shadow, and local-shadow indices.");
        return 1;
    }

    return 0;
}
