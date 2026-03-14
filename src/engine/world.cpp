#include "engine/world.h"

#include <algorithm>
#include <stdexcept>

namespace cressim::neo::engine
{

namespace
{

const std::vector<ColliderHandle>& emptyColliderHandleList()
{
    static const std::vector<ColliderHandle> kEmpty;
    return kEmpty;
}

} // namespace

common::EntityId World::createEntity()
{
    const common::EntityId entityId = mNextEntityId++;
    mAlive.insert(entityId);
    mEntities.push_back(entityId);
    markDirty(entityId);
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
    mRigidBodies.erase(entityId);
    removeCollidersForEntity(entityId);

    mEntities.erase(std::remove(mEntities.begin(), mEntities.end(), entityId), mEntities.end());
    markDirty(entityId);
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

TransformComponent& World::setTransform(common::EntityId entityId,
                                        const TransformComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setTransform requires a valid entity id.");
    }

    ensureEntity(entityId);
    TransformComponent& updated = mTransforms[entityId] = component;
    markDirty(entityId);
    return updated;
}

MeshRendererComponent& World::setMeshRenderer(common::EntityId entityId,
                                              const MeshRendererComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setMeshRenderer requires a valid entity id.");
    }

    ensureEntity(entityId);
    MeshRendererComponent& updated = mMeshRenderers[entityId] = component;
    markDirty(entityId);
    return updated;
}

CameraComponent& World::setCamera(common::EntityId entityId, const CameraComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setCamera requires a valid entity id.");
    }

    ensureEntity(entityId);
    CameraComponent& updated = mCameras[entityId] = component;
    markDirty(entityId);
    return updated;
}

DirectionalLightComponent& World::setDirectionalLight(common::EntityId entityId,
                                                      const DirectionalLightComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setDirectionalLight requires a valid entity id.");
    }

    ensureEntity(entityId);
    DirectionalLightComponent& updated = mDirectionalLights[entityId] = component;
    markDirty(entityId);
    return updated;
}

RigidBodyComponent& World::setRigidBody(common::EntityId entityId,
                                        const RigidBodyComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setRigidBody requires a valid entity id.");
    }

    ensureEntity(entityId);
    RigidBodyComponent& updated = mRigidBodies[entityId] = component;
    markDirty(entityId);
    return updated;
}

ColliderHandle World::addCollider(common::EntityId entityId, const ColliderComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("addCollider requires a valid entity id.");
    }

    ensureEntity(entityId);

    ColliderHandle handle{mNextColliderId++};
    mColliders.emplace(handle.id, ColliderRecord{entityId, component});
    mEntityColliderHandles[entityId].push_back(handle);
    markDirty(entityId);
    return handle;
}

ColliderComponent& World::updateCollider(ColliderHandle handle, const ColliderComponent& component)
{
    if (!handle.isValid())
    {
        throw std::invalid_argument("updateCollider requires a valid collider handle.");
    }

    const auto colliderIt = mColliders.find(handle.id);
    if (colliderIt == mColliders.end())
    {
        throw std::invalid_argument("updateCollider requires an existing collider handle.");
    }

    colliderIt->second.component = component;
    markDirty(colliderIt->second.ownerEntityId);
    return colliderIt->second.component;
}

bool World::removeTransform(common::EntityId entityId)
{
    if (mTransforms.erase(entityId) == 0)
    {
        return false;
    }
    markDirty(entityId);
    return true;
}

bool World::removeMeshRenderer(common::EntityId entityId)
{
    if (mMeshRenderers.erase(entityId) == 0)
    {
        return false;
    }
    markDirty(entityId);
    return true;
}

bool World::removeCamera(common::EntityId entityId)
{
    if (mCameras.erase(entityId) == 0)
    {
        return false;
    }
    markDirty(entityId);
    return true;
}

bool World::removeDirectionalLight(common::EntityId entityId)
{
    if (mDirectionalLights.erase(entityId) == 0)
    {
        return false;
    }
    markDirty(entityId);
    return true;
}

bool World::removeRigidBody(common::EntityId entityId)
{
    if (mRigidBodies.erase(entityId) == 0)
    {
        return false;
    }

    removeCollidersForEntity(entityId);
    markDirty(entityId);
    return true;
}

bool World::removeCollider(ColliderHandle handle)
{
    if (!handle.isValid())
    {
        return false;
    }

    const auto colliderIt = mColliders.find(handle.id);
    if (colliderIt == mColliders.end())
    {
        return false;
    }

    const common::EntityId ownerEntityId = colliderIt->second.ownerEntityId;
    auto handlesIt = mEntityColliderHandles.find(ownerEntityId);
    if (handlesIt != mEntityColliderHandles.end())
    {
        auto& handles = handlesIt->second;
        handles.erase(std::remove_if(handles.begin(), handles.end(),
                                     [handle](const ColliderHandle candidate) {
                                         return candidate.id == handle.id;
                                     }),
                      handles.end());
        if (handles.empty())
        {
            mEntityColliderHandles.erase(handlesIt);
        }
    }

    mColliders.erase(colliderIt);
    markDirty(ownerEntityId);
    return true;
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

const RigidBodyComponent* World::tryGetRigidBody(common::EntityId entityId) const
{
    const auto it = mRigidBodies.find(entityId);
    return it != mRigidBodies.end() ? &it->second : nullptr;
}

const ColliderComponent* World::tryGetCollider(ColliderHandle handle) const
{
    const auto it = mColliders.find(handle.id);
    return it != mColliders.end() ? &it->second.component : nullptr;
}

const std::vector<ColliderHandle>& World::colliderHandles(common::EntityId entityId) const
{
    const auto it = mEntityColliderHandles.find(entityId);
    return it != mEntityColliderHandles.end() ? it->second : emptyColliderHandleList();
}

std::uint64_t World::revision() const noexcept
{
    return mRevision;
}

const std::vector<common::EntityId>& World::dirtyEntities() const noexcept
{
    return mDirtyEntities;
}

void World::clearDirtyEntities() noexcept
{
    mDirtyEntities.clear();
    mDirtySet.clear();
}

void World::removeCollidersForEntity(common::EntityId entityId)
{
    const auto handlesIt = mEntityColliderHandles.find(entityId);
    if (handlesIt == mEntityColliderHandles.end())
    {
        return;
    }

    for (const ColliderHandle handle : handlesIt->second)
    {
        mColliders.erase(handle.id);
    }
    mEntityColliderHandles.erase(handlesIt);
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

void World::markDirty(common::EntityId entityId)
{
    ++mRevision;
    if (mDirtySet.insert(entityId).second)
    {
        mDirtyEntities.push_back(entityId);
    }
}

} // namespace cressim::neo::engine
