#include "engine/world_to_render_world_sync.h"

namespace cressim::neo::engine::detail
{

void syncWorldToRenderWorld(const World& world, graphics::RenderWorld& renderWorld)
{
    for (const common::EntityId entityId : world.renderDirtyEntities())
    {
        if (!world.isAlive(entityId))
        {
            // Remove from render world if not alive any more
            (void)renderWorld.removeRenderable(entityId);
            (void)renderWorld.removeCamera(entityId);
            (void)renderWorld.removeDirectionalLight(entityId);
            continue;
        }

        const std::optional<TransformComponent> transform = world.tryGetTransform(entityId);
        const common::Transform worldTransform = transform ? transform->worldTransform
                                                           : common::Transform{};

        const std::optional<MeshRendererComponent> meshRenderer = world.tryGetMeshRenderer(entityId);
        if (meshRenderer && meshRenderer->visible)
        {
            graphics::RenderableInstance renderable{};
            renderable.entityId       = entityId;
            renderable.worldTransform = worldTransform;
            renderable.mesh           = meshRenderer->mesh;
            renderable.material       = meshRenderer->material;
            renderWorld.upsertRenderable(renderable);
        }
        else
        {
            (void)renderWorld.removeRenderable(entityId);
        }

        const std::optional<CameraComponent> camera = world.tryGetCamera(entityId);
        if (camera)
        {
            graphics::CameraData cameraData{};
            cameraData.entityId           = entityId;
            cameraData.worldTransform     = worldTransform;
            cameraData.verticalFovDegrees = camera->verticalFovDegrees;
            cameraData.aspectRatio        = camera->aspectRatio;
            cameraData.nearClip           = camera->nearClip;
            cameraData.farClip            = camera->farClip;
            cameraData.outputTarget       = camera->outputTarget;
            cameraData.outputWidth        = camera->outputWidth;
            cameraData.outputHeight       = camera->outputHeight;
            cameraData.viewport           = camera->viewport;
            cameraData.renderOrder        = camera->renderOrder;
            renderWorld.upsertCamera(cameraData);
        }
        else
        {
            (void)renderWorld.removeCamera(entityId);
        }

        const std::optional<DirectionalLightComponent> directionalLight =
            world.tryGetDirectionalLight(entityId);
        if (directionalLight)
        {
            graphics::DirectionalLightData lightData{};
            lightData.entityId           = entityId;
            lightData.direction          = directionalLight->direction;
            lightData.color              = directionalLight->color;
            lightData.intensity          = directionalLight->intensity;
            lightData.shadowDistance     = directionalLight->shadowDistance;
            lightData.shadowFadeDistance = directionalLight->shadowFadeDistance;
            renderWorld.upsertDirectionalLight(lightData);
        }
        else
        {
            (void)renderWorld.removeDirectionalLight(entityId);
        }
    }
}

} // namespace cressim::neo::engine::detail
