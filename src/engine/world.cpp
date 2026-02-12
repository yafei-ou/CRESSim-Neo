#include "engine/world.h"

#include <algorithm>
#include <stdexcept>

namespace cressim::neo::engine
{

common::EntityId World::createEntity()
{
    const common::EntityId entityId = mNextEntityId++;
    mAlive.insert(entityId);
    mEntities.push_back(entityId);
    return entityId;
}

bool World::destroyEntity(common::EntityId entityId)
{
    if (mAlive.erase(entityId) == 0)
    {
        return false;
    }

    mTransforms.erase(entityId);
    mMeshRenderers.erase(entityId);
    mCameras.erase(entityId);
    mDirectionalLights.erase(entityId);

    mEntities.erase(std::remove(mEntities.begin(), mEntities.end(), entityId), mEntities.end());
    return true;
}

bool World::isAlive(common::EntityId entityId) const
{
    return mAlive.find(entityId) != mAlive.end();
}

const std::vector<common::EntityId>& World::entities() const noexcept
{
    return mEntities;
}

TransformComponent& World::setTransform(common::EntityId entityId, const TransformComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setTransform requires a valid entity id.");
    }

    ensureEntity(entityId);
    return mTransforms[entityId] = component;
}

MeshRendererComponent& World::setMeshRenderer(common::EntityId entityId, const MeshRendererComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setMeshRenderer requires a valid entity id.");
    }

    ensureEntity(entityId);
    return mMeshRenderers[entityId] = component;
}

CameraComponent& World::setCamera(common::EntityId entityId, const CameraComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setCamera requires a valid entity id.");
    }

    ensureEntity(entityId);
    return mCameras[entityId] = component;
}

DirectionalLightComponent& World::setDirectionalLight(common::EntityId entityId, const DirectionalLightComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setDirectionalLight requires a valid entity id.");
    }

    ensureEntity(entityId);
    return mDirectionalLights[entityId] = component;
}

const TransformComponent* World::tryGetTransform(common::EntityId entityId) const
{
    const auto it = mTransforms.find(entityId);
    return it != mTransforms.end() ? &it->second : nullptr;
}

const MeshRendererComponent* World::tryGetMeshRenderer(common::EntityId entityId) const
{
    const auto it = mMeshRenderers.find(entityId);
    return it != mMeshRenderers.end() ? &it->second : nullptr;
}

const CameraComponent* World::tryGetCamera(common::EntityId entityId) const
{
    const auto it = mCameras.find(entityId);
    return it != mCameras.end() ? &it->second : nullptr;
}

const DirectionalLightComponent* World::tryGetDirectionalLight(common::EntityId entityId) const
{
    const auto it = mDirectionalLights.find(entityId);
    return it != mDirectionalLights.end() ? &it->second : nullptr;
}

void World::ensureEntity(common::EntityId entityId)
{
    if (mAlive.find(entityId) != mAlive.end())
    {
        return;
    }

    mAlive.insert(entityId);
    mEntities.push_back(entityId);

    if (entityId >= mNextEntityId)
    {
        mNextEntityId = entityId + 1;
    }
}

} // namespace cressim::neo::engine
