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
    markRenderDirty(entityId);
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
    (void)mPhysicsWorld.removeRigidBody(entityId);
    removeCollidersForEntity(entityId);

    mEntities.erase(std::remove(mEntities.begin(), mEntities.end(), entityId), mEntities.end());
    markRenderDirty(entityId);
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
    syncRigidBodyToPhysics(entityId);
    const auto handlesIt = mEntityColliderHandles.find(entityId);
    if (handlesIt != mEntityColliderHandles.end())
    {
        for (const ColliderHandle handle : handlesIt->second)
        {
            syncColliderToPhysics(handle);
        }
    }
    markRenderDirty(entityId);
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
    markRenderDirty(entityId);
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
    markRenderDirty(entityId);
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
    markRenderDirty(entityId);
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
    if (!updated.simulated)
    {
        (void)mPhysicsWorld.removeRigidBody(entityId);
    }
    else
    {
        mTransforms.try_emplace(entityId, TransformComponent{});
        syncRigidBodyToPhysics(entityId);
        const auto handlesIt = mEntityColliderHandles.find(entityId);
        if (handlesIt != mEntityColliderHandles.end())
        {
            for (const ColliderHandle handle : handlesIt->second)
            {
                syncColliderToPhysics(handle);
            }
        }
    }
    markRenderDirty(entityId);
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
    syncColliderToPhysics(handle);
    markRenderDirty(entityId);
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
    syncColliderToPhysics(handle);
    markRenderDirty(colliderIt->second.ownerEntityId);
    return colliderIt->second.component;
}

bool World::removeTransform(common::EntityId entityId)
{
    if (mTransforms.erase(entityId) == 0)
    {
        return false;
    }
    (void)mPhysicsWorld.removeRigidBody(entityId);
    markRenderDirty(entityId);
    return true;
}

bool World::removeMeshRenderer(common::EntityId entityId)
{
    if (mMeshRenderers.erase(entityId) == 0)
    {
        return false;
    }
    markRenderDirty(entityId);
    return true;
}

bool World::removeCamera(common::EntityId entityId)
{
    if (mCameras.erase(entityId) == 0)
    {
        return false;
    }
    markRenderDirty(entityId);
    return true;
}

bool World::removeDirectionalLight(common::EntityId entityId)
{
    if (mDirectionalLights.erase(entityId) == 0)
    {
        return false;
    }
    markRenderDirty(entityId);
    return true;
}

bool World::removeRigidBody(common::EntityId entityId)
{
    if (mRigidBodies.erase(entityId) == 0)
    {
        return false;
    }

    (void)mPhysicsWorld.removeRigidBody(entityId);
    markRenderDirty(entityId);
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
    (void)mPhysicsWorld.removeCollider(handle.id);
    markRenderDirty(ownerEntityId);
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

physics::PhysicsWorld& World::physicsWorld() noexcept
{
    return mPhysicsWorld;
}

const physics::PhysicsWorld& World::physicsWorld() const noexcept
{
    return mPhysicsWorld;
}

void World::refreshFromPhysics()
{
    const physics::PhysicsSoADirtyRange& dirtyRange = mPhysicsWorld.rigidBodyDirtyRange();
    if (!dirtyRange.valid)
    {
        return;
    }

    const auto& states = mPhysicsWorld.rigidBodySnapshot();
    const std::uint32_t end =
        std::min<std::uint32_t>(dirtyRange.end, static_cast<std::uint32_t>(states.size()));
    for (std::uint32_t i = dirtyRange.begin; i < end; ++i)
    {
        const physics::RigidBodyState& state = states[i];
        if (!isAlive(state.entityId))
        {
            continue;
        }

        TransformComponent& transform = mTransforms[state.entityId];
        transform.worldTransform.position = state.position;
        transform.worldTransform.rotation = state.rotation;
        transform.worldTransform.scale    = state.scale;

        auto bodyIt = mRigidBodies.find(state.entityId);
        if (bodyIt != mRigidBodies.end())
        {
            RigidBodyComponent& rigidBody = bodyIt->second;
            rigidBody.bodyType                = state.bodyType;
            rigidBody.linearVelocity          = state.linearVelocity;
            rigidBody.angularVelocity         = state.angularVelocity;
            rigidBody.inverseMass             = state.inverseMass;
            rigidBody.inverseInertiaLocal     = state.inverseInertiaLocal;
            rigidBody.kinematicTargetPosition = state.kinematicTargetPosition;
            rigidBody.kinematicTargetRotation = state.kinematicTargetRotation;
            rigidBody.kinematicTargetEnabled  = state.kinematicTargetEnabled;
        }

        markRenderDirty(state.entityId);
    }
}

std::uint64_t World::renderRevision() const noexcept
{
    return mRenderRevision;
}

const std::vector<common::EntityId>& World::renderDirtyEntities() const noexcept
{
    return mRenderDirtyEntities;
}

void World::clearRenderDirtyEntities() noexcept
{
    mRenderDirtyEntities.clear();
    mRenderDirtySet.clear();
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
        (void)mPhysicsWorld.removeCollider(handle.id);
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

void World::markRenderDirty(common::EntityId entityId)
{
    ++mRenderRevision;
    if (mRenderDirtySet.insert(entityId).second)
    {
        mRenderDirtyEntities.push_back(entityId);
    }
}

void World::syncRigidBodyToPhysics(common::EntityId entityId)
{
    const auto transformIt = mTransforms.find(entityId);
    const auto rigidBodyIt = mRigidBodies.find(entityId);
    if (transformIt == mTransforms.end() || rigidBodyIt == mRigidBodies.end() ||
        !rigidBodyIt->second.simulated)
    {
        (void)mPhysicsWorld.removeRigidBody(entityId);
        return;
    }

    physics::RigidBodyState state{};
    state.entityId            = entityId;
    state.position            = transformIt->second.worldTransform.position;
    state.rotation            = transformIt->second.worldTransform.rotation;
    state.scale               = transformIt->second.worldTransform.scale;
    state.linearVelocity      = rigidBodyIt->second.linearVelocity;
    state.angularVelocity     = rigidBodyIt->second.angularVelocity;
    state.inverseInertiaLocal = rigidBodyIt->second.inverseInertiaLocal;
    state.bodyType            = rigidBodyIt->second.bodyType;
    state.inverseMass         = rigidBodyIt->second.inverseMass;
    state.kinematicTargetPosition = rigidBodyIt->second.kinematicTargetPosition;
    state.kinematicTargetRotation = rigidBodyIt->second.kinematicTargetRotation;
    state.kinematicTargetEnabled  = rigidBodyIt->second.kinematicTargetEnabled;
    (void)mPhysicsWorld.upsertRigidBody(state);
}

void World::syncColliderToPhysics(ColliderHandle handle)
{
    if (!handle.isValid())
    {
        return;
    }

    const auto colliderIt = mColliders.find(handle.id);
    if (colliderIt == mColliders.end())
    {
        (void)mPhysicsWorld.removeCollider(handle.id);
        return;
    }

    const auto bodyIt = mRigidBodies.find(colliderIt->second.ownerEntityId);
    const auto transformIt = mTransforms.find(colliderIt->second.ownerEntityId);
    if (bodyIt == mRigidBodies.end() || transformIt == mTransforms.end() || !bodyIt->second.simulated)
    {
        (void)mPhysicsWorld.removeCollider(handle.id);
        return;
    }

    physics::ColliderState state{};
    state.colliderId       = handle.id;
    state.entityId         = colliderIt->second.ownerEntityId;
    state.shapeType        = colliderIt->second.component.shapeType;
    state.shapeParams      = colliderIt->second.component.shapeParams;
    state.localPosition    = colliderIt->second.component.localPosition;
    state.localRotation    = colliderIt->second.component.localRotation;
    state.enabled          = colliderIt->second.component.enabled;
    state.friction         = colliderIt->second.component.friction;
    state.restitution      = colliderIt->second.component.restitution;
    state.collisionLayer   = colliderIt->second.component.collisionLayer;
    state.collisionMask    = colliderIt->second.component.collisionMask;
    mPhysicsWorld.upsertCollider(state);
}

} // namespace cressim::neo::engine
