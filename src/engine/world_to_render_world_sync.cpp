#include "engine/world_to_render_world_sync.h"

namespace cressim::neo::engine::detail
{

void syncWorldToRenderWorld(const World& world, graphics::RenderWorld& renderWorld)
{
    // Full rebuild each frame for now; straightforward but not optimal for large scenes.
    // TODO: switch to dirty-entity sync to avoid re-uploading unchanged data.
    renderWorld.clear();

    for (const common::EntityId entityId : world.entities())
    {
        const TransformComponent* transform = world.tryGetTransform(entityId);
        const common::Transform worldTransform = transform ? transform->worldTransform : common::Transform{};

        const MeshRendererComponent* meshRenderer = world.tryGetMeshRenderer(entityId);
        if (meshRenderer != nullptr && meshRenderer->visible)
        {
            graphics::RenderableInstance renderable{};
            renderable.entityId = entityId;
            renderable.worldTransform = worldTransform;
            renderable.mesh = meshRenderer->mesh;
            renderable.material = meshRenderer->material;
            renderWorld.upsertRenderable(renderable);
        }

        const CameraComponent* camera = world.tryGetCamera(entityId);
        if (camera != nullptr)
        {
            graphics::CameraData cameraData{};
            cameraData.entityId = entityId;
            cameraData.worldTransform = worldTransform;
            cameraData.verticalFovDegrees = camera->verticalFovDegrees;
            cameraData.nearClip = camera->nearClip;
            cameraData.farClip = camera->farClip;
            cameraData.outputTarget = camera->outputTarget;
            cameraData.outputWidth = camera->outputWidth;
            cameraData.outputHeight = camera->outputHeight;
            cameraData.viewport = camera->viewport;
            cameraData.renderOrder = camera->renderOrder;
            renderWorld.upsertCamera(cameraData);
        }

        const DirectionalLightComponent* directionalLight = world.tryGetDirectionalLight(entityId);
        if (directionalLight != nullptr)
        {
            graphics::DirectionalLightData lightData{};
            lightData.entityId = entityId;
            lightData.direction = directionalLight->direction;
            lightData.color = directionalLight->color;
            lightData.intensity = directionalLight->intensity;
            renderWorld.upsertDirectionalLight(lightData);
        }
    }
}

} // namespace cressim::neo::engine::detail
