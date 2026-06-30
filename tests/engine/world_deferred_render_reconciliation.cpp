#include "engine/components.h"
#include "engine/world.h"
#include "graphics/render_resource_manager.h"
#include "common/logger.h"

#include <cmath>

namespace
{

using cressim::neo::common::EntityId;
using cressim::neo::engine::CameraComponent;
using cressim::neo::engine::MeshRendererComponent;
using cressim::neo::engine::TransformComponent;
using cressim::neo::engine::World;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MeshResourceDesc;

bool nearlyEqual(float a, float b)
{
    return std::fabs(a - b) <= 1e-5f;
}

} // namespace

int main()
{
    using namespace cressim::neo;

    World world;
    world.setSceneLayout(common::SceneLayoutDesc{});

    graphics::RenderResourceManager resources;

    MeshResourceDesc meshDesc{};
    meshDesc.debugName = "DeferredRender.Mesh";
    meshDesc.vertices = {
        {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.0f, 0.0f},
        {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 1.0f, 0.0f},
        {{0.0f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5f, 1.0f}};
    meshDesc.indices = {0u, 1u, 2u};
    const auto mesh = resources.registerMesh(meshDesc);

    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "DeferredRender.Material";
    const auto material = resources.registerMaterial(materialDesc);

    const EntityId renderableEntity = world.createEntity();
    TransformComponent renderableTransform{};
    renderableTransform.worldTransform.position = {1.0f, 2.0f, 3.0f};
    world.setTransform(renderableEntity, renderableTransform);

    MeshRendererComponent renderer{};
    renderer.mesh = mesh;
    renderer.material = material;
    renderer.visible = true;
    world.setMeshRenderer(renderableEntity, renderer);

    const EntityId cameraEntity = world.createEntity();
    TransformComponent cameraTransform{};
    cameraTransform.worldTransform.position = {0.0f, 1.0f, -4.0f};
    world.setTransform(cameraEntity, cameraTransform);
    world.setCamera(cameraEntity, CameraComponent{});

    world.ensureRenderStateUpToDate(resources);

    const auto& positions0 = world.renderObjectPositions();
    if (positions0.empty() || !nearlyEqual(positions0[0].x, 1.0f) ||
        !nearlyEqual(positions0[0].y, 2.0f) || !nearlyEqual(positions0[0].z, 3.0f))
    {
        CRESSIM_LOG_ERROR( "Initial deferred pose flush failed.\n");
        return 1;
    }

    const auto& cameraInputs0 = world.cameraInputs();
    const auto& cameras0 = world.cameras();
    if (cameraInputs0.empty() || cameraInputs0[0].active == 0u || cameras0.empty() ||
        cameraInputs0[0].entityPoseSlot == 0xffffffffu ||
        !nearlyEqual(cameras0[0].worldTransform.position.y, 1.0f))
    {
        CRESSIM_LOG_ERROR( "Initial deferred camera flush failed.\n");
        return 1;
    }

    renderableTransform.worldTransform.position = {4.0f, 5.0f, 6.0f};
    world.setTransform(renderableEntity, renderableTransform);
    cameraTransform.worldTransform.position = {0.0f, 3.0f, -6.0f};
    world.setTransform(cameraEntity, cameraTransform);

    const auto& positionsBeforeFlush = world.renderObjectPositions();
    const auto& cameraInputsBeforeFlush = world.cameraInputs();
    const auto& camerasBeforeFlush = world.cameras();
    if (!nearlyEqual(positionsBeforeFlush[0].x, 1.0f) ||
        !nearlyEqual(camerasBeforeFlush[0].worldTransform.position.y, 1.0f))
    {
        CRESSIM_LOG_ERROR( "Derived render state updated eagerly instead of deferring.\n");
        return 1;
    }

    world.ensureRenderStateUpToDate(resources);

    const auto& positions1 = world.renderObjectPositions();
    const auto& cameraInputs1 = world.cameraInputs();
    const auto& cameras1 = world.cameras();
    if (!nearlyEqual(positions1[0].x, 4.0f) || !nearlyEqual(positions1[0].y, 5.0f) ||
        !nearlyEqual(positions1[0].z, 6.0f))
    {
        CRESSIM_LOG_ERROR( "Deferred renderable pose flush mismatch.\n");
        return 1;
    }
    if (cameraInputs1[0].entityPoseSlot == 0xffffffffu ||
        !nearlyEqual(cameras1[0].worldTransform.position.y, 3.0f) ||
        !nearlyEqual(cameras1[0].worldTransform.position.z, -6.0f))
    {
        CRESSIM_LOG_ERROR( "Deferred camera flush mismatch.\n");
        return 1;
    }

    CRESSIM_LOG_INFO( "Deferred render reconciliation checks passed.\n");
    return 0;
}
