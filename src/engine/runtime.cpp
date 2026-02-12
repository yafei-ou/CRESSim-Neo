#include "engine/runtime.h"

namespace cressim::neo::engine
{

bool Runtime::initialize(const RuntimeConfig& config)
{
    if (mInitialized)
    {
        return true;
    }

    mGraphicsDevice = graphics::createGraphicsDevice();
    if (!mGraphicsDevice)
    {
        return false;
    }

    if (!mGraphicsDevice->initialize(config.graphics))
    {
        mGraphicsDevice.reset();
        return false;
    }

    mRenderer = std::make_unique<graphics::Renderer>(*mGraphicsDevice, mScene.resources());
    if (!mRenderer->initialize())
    {
        mRenderer.reset();
        mGraphicsDevice->shutdown();
        mGraphicsDevice.reset();
        return false;
    }

    mInitialized = true;
    return true;
}

void Runtime::shutdown()
{
    if (!mInitialized)
    {
        return;
    }

    mRenderer.reset();

    if (mGraphicsDevice)
    {
        mGraphicsDevice->shutdown();
        mGraphicsDevice.reset();
    }

    mInitialized = false;
}

void Runtime::tick(const common::FrameContext& frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    syncWorldToRenderWorld();
    (void)mRenderer->render(frameContext, mScene.world());
}

World& Runtime::getWorld() noexcept
{
    return mWorld;
}

const World& Runtime::getWorld() const noexcept
{
    return mWorld;
}

graphics::Scene& Runtime::getScene() noexcept
{
    return mScene;
}

const graphics::Scene& Runtime::getScene() const noexcept
{
    return mScene;
}

void Runtime::syncWorldToRenderWorld()
{
    graphics::RenderWorld& renderWorld = mScene.world();
    renderWorld.clear();

    for (const common::EntityId entityId : mWorld.entities())
    {
        const TransformComponent* transform = mWorld.tryGetTransform(entityId);
        const common::Transform worldTransform = transform ? transform->worldTransform : common::Transform{};

        const MeshRendererComponent* meshRenderer = mWorld.tryGetMeshRenderer(entityId);
        if (meshRenderer != nullptr && meshRenderer->visible)
        {
            graphics::RenderableInstance renderable{};
            renderable.entityId = entityId;
            renderable.worldTransform = worldTransform;
            renderable.mesh = meshRenderer->mesh;
            renderable.material = meshRenderer->material;
            renderWorld.upsertRenderable(renderable);
        }

        const CameraComponent* camera = mWorld.tryGetCamera(entityId);
        if (camera != nullptr)
        {
            graphics::CameraData cameraData{};
            cameraData.entityId = entityId;
            cameraData.worldTransform = worldTransform;
            cameraData.verticalFovDegrees = camera->verticalFovDegrees;
            cameraData.nearClip = camera->nearClip;
            cameraData.farClip = camera->farClip;
            renderWorld.upsertCamera(cameraData);
        }

        const DirectionalLightComponent* directionalLight = mWorld.tryGetDirectionalLight(entityId);
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

} // namespace cressim::neo::engine
