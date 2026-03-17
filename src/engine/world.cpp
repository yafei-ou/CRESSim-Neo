#include "engine/world.h"

namespace cressim::neo::engine
{

namespace
{
const std::vector<World::ColliderHandle>& emptyColliderHandleList()
{
    static const std::vector<World::ColliderHandle> kEmpty;
    return kEmpty;
}
} // namespace

template <typename SoAType>
bool World::removeFromSoA(common::EntityId entityId, SoAType& soa, SparseIndex<SoAType>& index)
{
    const auto it = index.entityToIndex.find(entityId);
    if (it == index.entityToIndex.end())
    {
        return false;
    }

    const std::uint32_t removeIndex    = it->second;
    const std::uint32_t lastIndex      = static_cast<std::uint32_t>(soa.entityIds.size() - 1u);
    const common::EntityId movedEntity = soa.entityIds[lastIndex];

    if (removeIndex != lastIndex)
    {
        soa.entityIds[removeIndex]       = soa.entityIds[lastIndex];
        index.entityToIndex[movedEntity] = removeIndex;
    }

    soa.entityIds.pop_back();
    index.entityToIndex.erase(it);
    return true;
}

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

    removeTransform(entityId);
    removeMeshRenderer(entityId);
    removeCamera(entityId);
    removeDirectionalLight(entityId);
    removeRigidBody(entityId);

    auto physIt = mPhysicsLinks.find(entityId);
    if (physIt != mPhysicsLinks.end())
    {
        const auto colliders = physIt->second.colliders;
        for (const ColliderHandle h : colliders)
        {
            removeCollider(h);
        }
        mPhysicsLinks.erase(physIt);
    }

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

void World::setTransform(common::EntityId entityId, const TransformComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setTransform requires valid entity id.");
    }

    ensureEntity(entityId);

    upsertSoA(entityId, mTransforms, mTransformIndex,
              [&](std::uint32_t index, bool appended)
              {
                  if (appended)
                  {
                      mTransforms.positions.push_back(packPosition(component));
                      mTransforms.rotations.push_back(packRotation(component));
                      mTransforms.scales.push_back(packScale(component));
                  }
                  else
                  {
                      mTransforms.positions[index] = packPosition(component);
                      mTransforms.rotations[index] = packRotation(component);
                      mTransforms.scales[index]    = packScale(component);
                  }
              });

    // Update body pose directly in physics. No collider loop.
    if (auto* rb = mPhysicsWorld.tryGetRigidBody(entityId))
    {
        rb->position = component.worldTransform.position;
        rb->rotation = component.worldTransform.rotation;
        rb->scale    = component.worldTransform.scale;
        mPhysicsWorld.upsertRigidBody(*rb);
    }

    markRenderDirty(entityId);
}

void World::setMeshRenderer(common::EntityId entityId, const MeshRendererComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setMeshRenderer requires valid entity id.");
    }

    ensureEntity(entityId);

    upsertSoA(entityId, mMeshRenderers, mMeshRendererIndex,
              [&](std::uint32_t index, bool appended)
              {
                  if (appended)
                  {
                      mMeshRenderers.meshIds.push_back(component.mesh.id);
                      mMeshRenderers.materialIds.push_back(component.material.id);
                      mMeshRenderers.visibleFlags.push_back(component.visible ? 1u : 0u);
                  }
                  else
                  {
                      mMeshRenderers.meshIds[index]      = component.mesh.id;
                      mMeshRenderers.materialIds[index]  = component.material.id;
                      mMeshRenderers.visibleFlags[index] = component.visible ? 1u : 0u;
                  }
              });

    markRenderDirty(entityId);
}

void World::setCamera(common::EntityId entityId, const CameraComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setCamera requires valid entity id.");
    }

    ensureEntity(entityId);

    upsertSoA(entityId, mCameras, mCameraIndex,
              [&](std::uint32_t index, bool appended)
              {
                  if (appended)
                  {
                      mCameras.projection0.push_back(packCameraProjection0(component));
                      mCameras.projection1.push_back(packCameraProjection1(component));
                      mCameras.outputTargetIds.push_back(component.outputTarget.id);
                      mCameras.outputWidths.push_back(component.outputWidth);
                      mCameras.outputHeights.push_back(component.outputHeight);
                      mCameras.viewports.push_back(
                          Diligent::float4{component.viewport.x, component.viewport.y,
                                           component.viewport.width, component.viewport.height});
                      mCameras.renderOrders.push_back(component.renderOrder);
                  }
                  else
                  {
                      mCameras.projection0[index]     = packCameraProjection0(component);
                      mCameras.projection1[index]     = packCameraProjection1(component);
                      mCameras.outputTargetIds[index] = component.outputTarget.id;
                      mCameras.outputWidths[index]    = component.outputWidth;
                      mCameras.outputHeights[index]   = component.outputHeight;
                      mCameras.viewports[index] =
                          Diligent::float4{component.viewport.x, component.viewport.y,
                                           component.viewport.width, component.viewport.height};
                      mCameras.renderOrders[index] = component.renderOrder;
                  }
              });
    markRenderDirty(entityId);
}

void World::setDirectionalLight(common::EntityId entityId,
                                const DirectionalLightComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setDirectionalLight requires valid entity id.");
    }

    ensureEntity(entityId);

    upsertSoA(entityId, mDirectionalLights, mDirectionalLightIndex,
              [&](std::uint32_t index, bool appended)
              {
                  if (appended)
                  {
                      mDirectionalLights.directionsIntensities.push_back(
                          packLightDirectionIntensity(component));
                      mDirectionalLights.colors.push_back(packLightColor(component));
                      mDirectionalLights.shadowParams.push_back(packLightShadowParams(component));
                  }
                  else
                  {
                      mDirectionalLights.directionsIntensities[index] =
                          packLightDirectionIntensity(component);
                      mDirectionalLights.colors[index]       = packLightColor(component);
                      mDirectionalLights.shadowParams[index] = packLightShadowParams(component);
                  }
              });

    markRenderDirty(entityId);
}

void World::setRigidBody(common::EntityId entityId, const RigidBodyComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setRigidBody requires valid entity id.");
    }

    ensureEntity(entityId);

    if (!component.simulated)
    {
        mPhysicsWorld.removeRigidBody(entityId);
        mPhysicsLinks[entityId].hasRigidBody = false;
        markRenderDirty(entityId);
        return;
    }

    TransformComponent transform{};
    if (const std::optional<TransformComponent> t = tryGetTransform(entityId))
    {
        transform = *t;
    }

    physics::RigidBodyState state{};
    state.entityId                = entityId;
    state.position                = transform.worldTransform.position;
    state.rotation                = transform.worldTransform.rotation;
    state.scale                   = transform.worldTransform.scale;
    state.linearVelocity          = component.linearVelocity;
    state.angularVelocity         = component.angularVelocity;
    state.inverseMass             = component.inverseMass;
    state.inverseInertiaLocal     = component.inverseInertiaLocal;
    state.bodyType                = component.bodyType;
    state.kinematicTargetPosition = component.kinematicTargetPosition;
    state.kinematicTargetRotation = component.kinematicTargetRotation;
    state.kinematicTargetEnabled  = component.kinematicTargetEnabled;

    mPhysicsWorld.upsertRigidBody(state);
    mPhysicsLinks[entityId].hasRigidBody = true;
    markRenderDirty(entityId);
}

bool World::removeRigidBody(common::EntityId entityId)
{
    auto it = mPhysicsLinks.find(entityId);
    if (it != mPhysicsLinks.end())
    {
        it->second.hasRigidBody = false;
    }

    const bool removed = mPhysicsWorld.removeRigidBody(entityId);
    if (removed)
    {
        markRenderDirty(entityId);
    }
    return removed;
}

World::ColliderHandle World::addCollider(common::EntityId entityId,
                                         const ColliderComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("addCollider requires valid entity id.");
    }

    ensureEntity(entityId);

    auto body = mPhysicsLinks.find(entityId);
    if (body == mPhysicsLinks.end() || !body->second.hasRigidBody)
    {
        throw std::logic_error("addCollider requires a rigid body on the entity.");
    }

    ColliderHandle handle{mNextColliderId++};

    physics::ColliderState state{};
    state.colliderId     = handle.id;
    state.entityId       = entityId;
    state.shapeType      = component.shapeType;
    state.shapeParams    = component.shapeParams;
    state.localPosition  = component.localPosition;
    state.localRotation  = component.localRotation;
    state.enabled        = component.enabled;
    state.friction       = component.friction;
    state.restitution    = component.restitution;
    state.collisionLayer = component.collisionLayer;
    state.collisionMask  = component.collisionMask;

    mPhysicsWorld.upsertCollider(state);
    mPhysicsLinks[entityId].colliders.push_back(handle);
    mColliderOwnerEntity[handle.id] = entityId;
    markRenderDirty(entityId);
    return handle;
}

void World::updateCollider(ColliderHandle handle, const ColliderComponent& component)
{
    if (!handle.isValid())
    {
        throw std::invalid_argument("updateCollider requires valid collider handle.");
    }

    const auto ownerIt = mColliderOwnerEntity.find(handle.id);
    if (ownerIt == mColliderOwnerEntity.end())
    {
        throw std::invalid_argument("Unknown collider handle.");
    }

    const common::EntityId entityId = ownerIt->second;

    physics::ColliderState state{};
    state.colliderId     = handle.id;
    state.entityId       = entityId;
    state.shapeType      = component.shapeType;
    state.shapeParams    = component.shapeParams;
    state.localPosition  = component.localPosition;
    state.localRotation  = component.localRotation;
    state.enabled        = component.enabled;
    state.friction       = component.friction;
    state.restitution    = component.restitution;
    state.collisionLayer = component.collisionLayer;
    state.collisionMask  = component.collisionMask;

    mPhysicsWorld.upsertCollider(state);
    markRenderDirty(entityId);
}

bool World::removeCollider(ColliderHandle handle)
{
    if (!handle.isValid())
    {
        return false;
    }

    const auto ownerIt = mColliderOwnerEntity.find(handle.id);
    if (ownerIt == mColliderOwnerEntity.end())
    {
        return false;
    }

    const common::EntityId entityId = ownerIt->second;
    auto physIt                     = mPhysicsLinks.find(entityId);
    if (physIt != mPhysicsLinks.end())
    {
        auto& handles = physIt->second.colliders;
        handles.erase(std::remove_if(handles.begin(), handles.end(),
                                     [&](const ColliderHandle h) { return h.id == handle.id; }),
                      handles.end());
    }

    mColliderOwnerEntity.erase(ownerIt);
    const bool removed = mPhysicsWorld.removeCollider(handle.id);
    if (removed)
    {
        markRenderDirty(entityId);
    }
    return removed;
}

bool World::removeTransform(common::EntityId entityId)
{
    const auto it = mTransformIndex.entityToIndex.find(entityId);
    if (it == mTransformIndex.entityToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mTransforms.entityIds.size() - 1u);
    const common::EntityId movedEntity = mTransforms.entityIds[last];

    if (index != last)
    {
        mTransforms.entityIds[index]               = mTransforms.entityIds[last];
        mTransforms.positions[index]               = mTransforms.positions[last];
        mTransforms.rotations[index]               = mTransforms.rotations[last];
        mTransforms.scales[index]                  = mTransforms.scales[last];
        mTransformIndex.entityToIndex[movedEntity] = index;
    }

    mTransforms.entityIds.pop_back();
    mTransforms.positions.pop_back();
    mTransforms.rotations.pop_back();
    mTransforms.scales.pop_back();
    mTransformIndex.entityToIndex.erase(it);

    markRenderDirty(entityId);
    return true;
}

bool World::removeMeshRenderer(common::EntityId entityId)
{
    const auto it = mMeshRendererIndex.entityToIndex.find(entityId);
    if (it == mMeshRendererIndex.entityToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mMeshRenderers.entityIds.size() - 1u);
    const common::EntityId movedEntity = mMeshRenderers.entityIds[last];

    if (index != last)
    {
        mMeshRenderers.entityIds[index]               = mMeshRenderers.entityIds[last];
        mMeshRenderers.meshIds[index]                 = mMeshRenderers.meshIds[last];
        mMeshRenderers.materialIds[index]             = mMeshRenderers.materialIds[last];
        mMeshRenderers.visibleFlags[index]            = mMeshRenderers.visibleFlags[last];
        mMeshRendererIndex.entityToIndex[movedEntity] = index;
    }

    mMeshRenderers.entityIds.pop_back();
    mMeshRenderers.meshIds.pop_back();
    mMeshRenderers.materialIds.pop_back();
    mMeshRenderers.visibleFlags.pop_back();
    mMeshRendererIndex.entityToIndex.erase(it);
    markRenderDirty(entityId);
    return true;
}

bool World::removeCamera(common::EntityId entityId)
{
    const auto it = mCameraIndex.entityToIndex.find(entityId);
    if (it == mCameraIndex.entityToIndex.end())
    {
        return false;
    }

    const std::uint32_t index          = it->second;
    const std::uint32_t last           = static_cast<std::uint32_t>(mCameras.entityIds.size() - 1u);
    const common::EntityId movedEntity = mCameras.entityIds[last];

    if (index != last)
    {
        mCameras.entityIds[index]               = mCameras.entityIds[last];
        mCameras.projection0[index]             = mCameras.projection0[last];
        mCameras.projection1[index]             = mCameras.projection1[last];
        mCameras.outputTargetIds[index]         = mCameras.outputTargetIds[last];
        mCameras.outputWidths[index]            = mCameras.outputWidths[last];
        mCameras.outputHeights[index]           = mCameras.outputHeights[last];
        mCameras.viewports[index]               = mCameras.viewports[last];
        mCameras.renderOrders[index]            = mCameras.renderOrders[last];
        mCameraIndex.entityToIndex[movedEntity] = index;
    }

    mCameras.entityIds.pop_back();
    mCameras.projection0.pop_back();
    mCameras.projection1.pop_back();
    mCameras.outputTargetIds.pop_back();
    mCameras.outputWidths.pop_back();
    mCameras.outputHeights.pop_back();
    mCameras.viewports.pop_back();
    mCameras.renderOrders.pop_back();
    mCameraIndex.entityToIndex.erase(it);
    markRenderDirty(entityId);
    return true;
}

bool World::removeDirectionalLight(common::EntityId entityId)
{
    const auto it = mDirectionalLightIndex.entityToIndex.find(entityId);
    if (it == mDirectionalLightIndex.entityToIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last = static_cast<std::uint32_t>(mDirectionalLights.entityIds.size() - 1u);
    const common::EntityId movedEntity = mDirectionalLights.entityIds[last];

    if (index != last)
    {
        mDirectionalLights.entityIds[index] = mDirectionalLights.entityIds[last];
        mDirectionalLights.directionsIntensities[index] =
            mDirectionalLights.directionsIntensities[last];
        mDirectionalLights.colors[index]                  = mDirectionalLights.colors[last];
        mDirectionalLights.shadowParams[index]            = mDirectionalLights.shadowParams[last];
        mDirectionalLightIndex.entityToIndex[movedEntity] = index;
    }

    mDirectionalLights.entityIds.pop_back();
    mDirectionalLights.directionsIntensities.pop_back();
    mDirectionalLights.colors.pop_back();
    mDirectionalLights.shadowParams.pop_back();
    mDirectionalLightIndex.entityToIndex.erase(it);
    markRenderDirty(entityId);
    return true;
}

std::optional<TransformComponent> World::tryGetTransform(common::EntityId entityId) const
{
    const auto it = mTransformIndex.entityToIndex.find(entityId);
    if (it == mTransformIndex.entityToIndex.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = it->second;
    return unpackTransform(mTransforms.positions[index], mTransforms.rotations[index],
                           mTransforms.scales[index]);
}

std::optional<MeshRendererComponent> World::tryGetMeshRenderer(common::EntityId entityId) const
{
    const auto it = mMeshRendererIndex.entityToIndex.find(entityId);
    if (it == mMeshRendererIndex.entityToIndex.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = it->second;
    MeshRendererComponent component{};
    component.mesh.id     = mMeshRenderers.meshIds[index];
    component.material.id = mMeshRenderers.materialIds[index];
    component.visible     = mMeshRenderers.visibleFlags[index] != 0u;
    return component;
}

std::optional<CameraComponent> World::tryGetCamera(common::EntityId entityId) const
{
    const auto it = mCameraIndex.entityToIndex.find(entityId);
    if (it == mCameraIndex.entityToIndex.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = it->second;
    return unpackCamera(mCameras.projection0[index], mCameras.outputTargetIds[index],
                        mCameras.outputWidths[index], mCameras.outputHeights[index],
                        mCameras.viewports[index], mCameras.renderOrders[index]);
}

std::optional<DirectionalLightComponent> World::tryGetDirectionalLight(
    common::EntityId entityId) const
{
    const auto it = mDirectionalLightIndex.entityToIndex.find(entityId);
    if (it == mDirectionalLightIndex.entityToIndex.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = it->second;
    return unpackDirectionalLight(mDirectionalLights.directionsIntensities[index],
                                  mDirectionalLights.colors[index],
                                  mDirectionalLights.shadowParams[index]);
}

std::optional<RigidBodyComponent> World::tryGetRigidBody(common::EntityId entityId) const
{
    const physics::RigidBodyState* rb = mPhysicsWorld.tryGetRigidBody(entityId);
    if (!rb)
    {
        return std::nullopt;
    }

    RigidBodyComponent component{};
    component.simulated               = true;
    component.bodyType                = rb->bodyType;
    component.linearVelocity          = rb->linearVelocity;
    component.angularVelocity         = rb->angularVelocity;
    component.inverseMass             = rb->inverseMass;
    component.inverseInertiaLocal     = rb->inverseInertiaLocal;
    component.kinematicTargetPosition = rb->kinematicTargetPosition;
    component.kinematicTargetRotation = rb->kinematicTargetRotation;
    component.kinematicTargetEnabled  = rb->kinematicTargetEnabled;
    return component;
}

std::optional<ColliderComponent> World::tryGetCollider(ColliderHandle handle) const
{
    if (!handle.isValid())
    {
        return std::nullopt;
    }

    const physics::ColliderState* c = mPhysicsWorld.tryGetCollider(handle.id);
    if (!c)
    {
        return std::nullopt;
    }

    ColliderComponent out{};
    out.shapeType      = c->shapeType;
    out.shapeParams    = c->shapeParams;
    out.localPosition  = c->localPosition;
    out.localRotation  = c->localRotation;
    out.enabled        = c->enabled;
    out.friction       = c->friction;
    out.restitution    = c->restitution;
    out.collisionLayer = c->collisionLayer;
    out.collisionMask  = c->collisionMask;
    return out;
}

const std::vector<World::ColliderHandle>& World::colliderHandles(common::EntityId entityId) const
{
    const auto it = mPhysicsLinks.find(entityId);
    return it != mPhysicsLinks.end() ? it->second.colliders : emptyColliderHandleList();
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
    // TODO: this is still not ideal
    // We could add a GPU pass to write directly to a GPU Entity transform buffer.
    // We can avoid GPU rb position readback completely and CPU Entity transform
    // buffer is only on-demand after rendering directly reads the GPU buffer
    // without entity uploading at all, and after GPU frustum culling is done.

    const physics::PhysicsSoADirtyRange& dirty = mPhysicsWorld.rigidBodyDirtyRange();
    if (!dirty.valid)
    {
        return;
    }

    const auto& states = mPhysicsWorld.rigidBodySnapshot();
    const std::uint32_t end =
        std::min<std::uint32_t>(dirty.end, static_cast<std::uint32_t>(states.size()));

    for (std::uint32_t i = dirty.begin; i < end; ++i)
    {
        const physics::RigidBodyState& rb = states[i];
        if (!isAlive(rb.entityId))
        {
            continue;
        }

        TransformComponent t{};
        t.worldTransform.position = rb.position;
        t.worldTransform.rotation = rb.rotation;
        t.worldTransform.scale    = rb.scale;
        setTransform(rb.entityId, t);
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

} // namespace cressim::neo::engine
