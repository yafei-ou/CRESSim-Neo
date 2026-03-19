#include "engine/world.h"

#include <map>

namespace cressim::neo::engine
{

namespace
{
constexpr std::uint32_t kInvalidSlot         = 0xffffffffu;
constexpr std::uint32_t kInvalidCommandIndex = 0xffffffffu;

struct DrawBucketKey
{
    graphics::MaterialProgramFamily programFamily = graphics::MaterialProgramFamily::StandardLit;
    std::uint32_t materialFeatureFlags            = 0u;
    common::ResourceId materialId                 = common::kInvalidResourceId;
    common::ResourceId meshId                     = common::kInvalidResourceId;

    [[nodiscard]] bool operator<(const DrawBucketKey& rhs) const noexcept
    {
        if (programFamily != rhs.programFamily)
        {
            return static_cast<std::uint32_t>(programFamily) <
                   static_cast<std::uint32_t>(rhs.programFamily);
        }
        if (materialFeatureFlags != rhs.materialFeatureFlags)
        {
            return materialFeatureFlags < rhs.materialFeatureFlags;
        }
        if (materialId != rhs.materialId)
        {
            return materialId < rhs.materialId;
        }
        return meshId < rhs.meshId;
    }
};

const std::vector<World::ColliderHandle>& emptyColliderHandleList()
{
    static const std::vector<World::ColliderHandle> kEmpty;
    return kEmpty;
}

std::uint32_t allocateDenseSlot(
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& freeSlotsByEnv,
    std::unordered_map<std::uint32_t, std::uint32_t>& nextSlotByEnv, std::uint32_t envIndex,
    std::uint32_t maxSlotsPerEnv, const char* slotType)
{
    auto& freeSlots = freeSlotsByEnv[envIndex];
    if (!freeSlots.empty())
    {
        const std::uint32_t slot = freeSlots.back();
        freeSlots.pop_back();
        return slot;
    }

    std::uint32_t& nextSlot = nextSlotByEnv[envIndex];
    if (nextSlot >= maxSlotsPerEnv)
    {
        throw std::overflow_error(std::string(slotType) + " capacity exceeded for environment.");
    }
    return nextSlot++;
}

void reclaimDenseSlot(std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& freeSlotsByEnv,
                      std::uint32_t envIndex, std::uint32_t slot)
{
    if (slot == kInvalidSlot)
    {
        return;
    }
    freeSlotsByEnv[envIndex].push_back(slot);
}

void enqueueDenseDirtyIndex(std::uint32_t index, std::vector<std::uint32_t>& dirtyIndices,
                            std::vector<std::uint8_t>& dirtyBits)
{
    if (index >= dirtyBits.size() || dirtyBits[index] != 0u)
    {
        return;
    }

    dirtyBits[index] = 1u;
    dirtyIndices.push_back(index);
}

} // namespace

common::EntityId World::createEntity(std::uint32_t envIndex)
{
    // A globally unique EntityId within this World.

    if (envIndex >= mSceneLayout.envCount)
    {
        throw std::out_of_range("createEntity envIndex exceeds configured environment count.");
    }
    const common::EntityId entityId = mNextEntityId++;
    mAlive.insert(entityId);
    mEntities.push_back(entityId);
    mEntityEnvironments[entityId] = envIndex;
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
    mEntityEnvironments.erase(entityId);
    return true;
}

void World::setSceneLayout(const gpu::GpuSceneLayoutDesc& layout)
{
    mSceneLayout = layout;
    ensureHostSceneStorage();
}

const gpu::GpuSceneLayoutDesc& World::sceneLayout() const noexcept
{
    return mSceneLayout;
}

bool World::setEntityEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    if (!isAlive(entityId))
    {
        return false;
    }
    if (envIndex >= mSceneLayout.envCount)
    {
        throw std::out_of_range(
            "setEntityEnvironment envIndex exceeds configured environment count.");
    }

    const std::uint32_t previousEnv = entityEnvironment(entityId);
    if (previousEnv == envIndex)
    {
        return true;
    }

    mEntityEnvironments[entityId] = envIndex;
    const auto physIt             = mPhysicsLinks.find(entityId);
    if (physIt != mPhysicsLinks.end())
    {
        for (const ColliderHandle handle : physIt->second.colliders)
        {
            const physics::ColliderState* existing = mPhysicsWorld.tryGetCollider(handle.id);
            if (existing == nullptr)
            {
                continue;
            }

            physics::ColliderState updated = *existing;
            updated.environmentIndex       = envIndex;
            mPhysicsWorld.upsertCollider(updated);
        }
    }
    moveRenderableToEnvironment(entityId, envIndex);
    moveCameraToEnvironment(entityId, envIndex);
    moveDirectionalLightToEnvironment(entityId, envIndex);
    mDrawRegistryDirty              = true;
    mPhysicsRenderableMappingsDirty = true;
    return true;
}

std::uint32_t World::entityEnvironment(common::EntityId entityId) const noexcept
{
    const auto it = mEntityEnvironments.find(entityId);
    return it != mEntityEnvironments.end() ? it->second : 0u;
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
    mEntityEnvironments.try_emplace(entityId, 0u);
    if (entityId >= mNextEntityId)
    {
        mNextEntityId = entityId + 1;
    }
}

void World::ensureHostSceneStorage()
{
    const std::size_t objectCapacity = mSceneLayout.totalObjectCapacity();
    if (mRenderables.size() != objectCapacity)
    {
        mRenderables.assign(objectCapacity, graphics::RenderableInstance{});
        mRenderObjectPositions.assign(objectCapacity, Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
        mRenderObjectOrientations.assign(objectCapacity, Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f});
        mRenderObjectScales.assign(objectCapacity, Diligent::float4{1.0f, 1.0f, 1.0f, 0.0f});
        mRenderableMetadataHost.assign(objectCapacity, gpu::GpuRenderableMetadata{});
        mRenderableQueueInfoHost.assign(objectCapacity, gpu::GpuRenderableQueueInfo{});
        mOpaqueDrawRegistryHost.clear();
        mShadowDrawRegistryHost.clear();
        mDirtyRenderablePoseIndices.clear();
        mDirtyRenderableMetadataIndices.clear();
        mDirtyRenderablePoseBits.assign(objectCapacity, 0u);
        mDirtyRenderableMetadataBits.assign(objectCapacity, 0u);
        mDirtyRenderablePoseIndices.reserve(objectCapacity);
        mDirtyRenderableMetadataIndices.reserve(objectCapacity);
        for (std::uint32_t i = 0; i < objectCapacity; ++i)
        {
            mDirtyRenderablePoseIndices.push_back(i);
            mDirtyRenderableMetadataIndices.push_back(i);
            mDirtyRenderablePoseBits[i]     = 1u;
            mDirtyRenderableMetadataBits[i] = 1u;
        }
        mDrawRegistryDirty              = true;
        mPhysicsRenderableMappingsDirty = true;
    }

    const std::size_t cameraCapacity = mSceneLayout.totalCameraCapacity();
    if (mRenderCameras.size() != cameraCapacity)
    {
        mRenderCameras.assign(cameraCapacity, graphics::CameraData{});
        mCameraInputsHost.assign(cameraCapacity, gpu::GpuCameraInput{});
        mDirtyCameraIndices.clear();
        mDirtyCameraBits.assign(cameraCapacity, 0u);
        mDirtyCameraIndices.reserve(cameraCapacity);
        for (std::uint32_t i = 0; i < cameraCapacity; ++i)
        {
            mDirtyCameraIndices.push_back(i);
            mDirtyCameraBits[i] = 1u;
        }
    }

    const std::size_t lightCapacity = mSceneLayout.totalLightCapacity();
    if (mRenderDirectionalLights.size() != lightCapacity)
    {
        mRenderDirectionalLights.assign(lightCapacity, graphics::DirectionalLightData{});
        mLightInputsHost.assign(lightCapacity, gpu::GpuDirectionalLightInput{});
        mDirtyLightIndices.clear();
        mDirtyLightBits.assign(lightCapacity, 0u);
        mDirtyLightIndices.reserve(lightCapacity);
        for (std::uint32_t i = 0; i < lightCapacity; ++i)
        {
            mDirtyLightIndices.push_back(i);
            mDirtyLightBits[i] = 1u;
        }
    }
}

void World::markRenderableMetadataDirty(std::uint32_t objectIndex)
{
    enqueueDenseDirtyIndex(objectIndex, mDirtyRenderableMetadataIndices,
                           mDirtyRenderableMetadataBits);
}

void World::markRenderablePoseDirty(std::uint32_t objectIndex)
{
    enqueueDenseDirtyIndex(objectIndex, mDirtyRenderablePoseIndices, mDirtyRenderablePoseBits);
}

void World::markCameraDirty(std::uint32_t cameraIndex)
{
    enqueueDenseDirtyIndex(cameraIndex, mDirtyCameraIndices, mDirtyCameraBits);
}

void World::markLightDirty(std::uint32_t lightIndex)
{
    enqueueDenseDirtyIndex(lightIndex, mDirtyLightIndices, mDirtyLightBits);
}

void World::clearDirtyIndexSet(std::vector<std::uint32_t>& dirtyIndices,
                               std::vector<std::uint8_t>& dirtyBits)
{
    for (const std::uint32_t index : dirtyIndices)
    {
        if (index < dirtyBits.size())
        {
            dirtyBits[index] = 0u;
        }
    }
    dirtyIndices.clear();
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
    if (const auto it = mRenderableIndices.find(entityId); it != mRenderableIndices.end())
    {
        markRenderablePoseDirty(static_cast<std::uint32_t>(it->second));
    }
    if (const auto it = mRenderCameraIndices.find(entityId); it != mRenderCameraIndices.end())
    {
        markCameraDirty(static_cast<std::uint32_t>(it->second));
    }
}

void World::setMeshRenderer(common::EntityId entityId, const MeshRendererComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setMeshRenderer requires valid entity id.");
    }

    ensureEntity(entityId);
    ensureHostSceneStorage();

    const auto indexIt        = mRenderableIndices.find(entityId);
    std::uint32_t objectIndex = 0u;
    if (indexIt == mRenderableIndices.end())
    {
        const std::uint32_t envIndex = entityEnvironment(entityId);
        const std::uint32_t objectSlot =
            allocateDenseSlot(mFreeRenderableSlotsByEnv, mNextRenderableSlotByEnv, envIndex,
                              mSceneLayout.maxObjectsPerEnv, "renderable");
        objectIndex                        = envIndex * mSceneLayout.maxObjectsPerEnv + objectSlot;
        mRenderableIndices[entityId]       = objectIndex;
        mRenderables[objectIndex].entityId = entityId;
        mRenderables[objectIndex].envIndex = envIndex;
        mRenderables[objectIndex].objectSlot = objectSlot;
    }
    else
    {
        objectIndex = static_cast<std::uint32_t>(indexIt->second);
    }

    graphics::RenderableInstance& renderable = mRenderables[objectIndex];
    renderable.mesh                          = component.mesh;
    renderable.material                      = component.material;
    renderable.visible                       = component.visible;
    markRenderableMetadataDirty(objectIndex);
    markRenderablePoseDirty(objectIndex);
    mDrawRegistryDirty              = true;
    mPhysicsRenderableMappingsDirty = true;
}

void World::setCamera(common::EntityId entityId, const CameraComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setCamera requires valid entity id.");
    }

    ensureEntity(entityId);
    ensureHostSceneStorage();

    const auto indexIt        = mRenderCameraIndices.find(entityId);
    std::uint32_t cameraIndex = 0u;
    if (indexIt == mRenderCameraIndices.end())
    {
        const std::uint32_t envIndex = entityEnvironment(entityId);
        const std::uint32_t cameraSlot =
            allocateDenseSlot(mFreeCameraSlotsByEnv, mNextCameraSlotByEnv, envIndex,
                              mSceneLayout.maxCamerasPerEnv, "camera");
        cameraIndex                    = envIndex * mSceneLayout.maxCamerasPerEnv + cameraSlot;
        mRenderCameraIndices[entityId] = cameraIndex;
        mRenderCameras[cameraIndex].entityId   = entityId;
        mRenderCameras[cameraIndex].envIndex   = envIndex;
        mRenderCameras[cameraIndex].cameraSlot = cameraSlot;
    }
    else
    {
        cameraIndex = static_cast<std::uint32_t>(indexIt->second);
    }

    graphics::CameraData& cameraData = mRenderCameras[cameraIndex];
    cameraData.worldTransform =
        tryGetTransform(entityId).value_or(TransformComponent{}).worldTransform;
    cameraData.verticalFovDegrees = component.verticalFovDegrees;
    cameraData.nearClip           = component.nearClip;
    cameraData.farClip            = component.farClip;
    cameraData.outputTarget       = component.outputTarget;
    cameraData.outputWidth        = component.outputWidth;
    cameraData.outputHeight       = component.outputHeight;
    cameraData.viewport           = component.viewport;
    cameraData.clearColor         = component.clearColor;
    cameraData.clearDepth         = component.clearDepth;
    cameraData.clearColorValue    = component.clearColorValue;
    cameraData.clearDepthValue    = component.clearDepthValue;
    cameraData.renderOrder        = component.renderOrder;
    markCameraDirty(cameraIndex);
}

void World::setDirectionalLight(common::EntityId entityId,
                                const DirectionalLightComponent& component)
{
    if (entityId == common::kInvalidEntityId)
    {
        throw std::invalid_argument("setDirectionalLight requires valid entity id.");
    }

    ensureEntity(entityId);
    ensureHostSceneStorage();

    const auto indexIt       = mRenderDirectionalLightIndices.find(entityId);
    std::uint32_t lightIndex = 0u;
    if (indexIt == mRenderDirectionalLightIndices.end())
    {
        const std::uint32_t envIndex = entityEnvironment(entityId);
        const std::uint32_t lightSlot =
            allocateDenseSlot(mFreeDirectionalLightSlotsByEnv, mNextDirectionalLightSlotByEnv,
                              envIndex, mSceneLayout.maxLightsPerEnv, "directional light");
        lightIndex = envIndex * mSceneLayout.maxLightsPerEnv + lightSlot;
        mRenderDirectionalLightIndices[entityId]       = lightIndex;
        mRenderDirectionalLights[lightIndex].entityId  = entityId;
        mRenderDirectionalLights[lightIndex].envIndex  = envIndex;
        mRenderDirectionalLights[lightIndex].lightSlot = lightSlot;
    }
    else
    {
        lightIndex = static_cast<std::uint32_t>(indexIt->second);
    }

    graphics::DirectionalLightData& lightData = mRenderDirectionalLights[lightIndex];
    lightData.direction                       = component.direction;
    lightData.color                           = component.color;
    lightData.intensity                       = component.intensity;
    lightData.shadowDistance                  = component.shadowDistance;
    lightData.shadowFadeDistance              = component.shadowFadeDistance;
    markLightDirty(lightIndex);
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
        mPhysicsRenderableMappingsDirty      = true;
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
    mPhysicsRenderableMappingsDirty      = true;
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
        mPhysicsRenderableMappingsDirty = true;
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
    state.colliderId       = handle.id;
    state.entityId         = entityId;
    state.shapeType        = component.shapeType;
    state.shapeParams      = component.shapeParams;
    state.localPosition    = component.localPosition;
    state.localRotation    = component.localRotation;
    state.environmentIndex = entityEnvironment(entityId);
    state.enabled          = component.enabled;
    state.friction         = component.friction;
    state.restitution      = component.restitution;
    state.collisionLayer   = component.collisionLayer;
    state.collisionMask    = component.collisionMask;

    mPhysicsWorld.upsertCollider(state);
    mPhysicsLinks[entityId].colliders.push_back(handle);
    mColliderOwnerEntity[handle.id] = entityId;
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
    state.colliderId       = handle.id;
    state.entityId         = entityId;
    state.shapeType        = component.shapeType;
    state.shapeParams      = component.shapeParams;
    state.localPosition    = component.localPosition;
    state.localRotation    = component.localRotation;
    state.environmentIndex = entityEnvironment(entityId);
    state.enabled          = component.enabled;
    state.friction         = component.friction;
    state.restitution      = component.restitution;
    state.collisionLayer   = component.collisionLayer;
    state.collisionMask    = component.collisionMask;

    mPhysicsWorld.upsertCollider(state);
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
    return mPhysicsWorld.removeCollider(handle.id);
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

    if (const auto it = mRenderableIndices.find(entityId); it != mRenderableIndices.end())
    {
        markRenderablePoseDirty(static_cast<std::uint32_t>(it->second));
    }
    if (const auto it = mRenderCameraIndices.find(entityId); it != mRenderCameraIndices.end())
    {
        markCameraDirty(static_cast<std::uint32_t>(it->second));
    }
    return true;
}

bool World::removeMeshRenderer(common::EntityId entityId)
{
    const auto it = mRenderableIndices.find(entityId);
    if (it == mRenderableIndices.end())
    {
        return false;
    }

    const std::uint32_t objectIndex              = static_cast<std::uint32_t>(it->second);
    const graphics::RenderableInstance& instance = mRenderables[objectIndex];
    reclaimDenseSlot(mFreeRenderableSlotsByEnv, instance.envIndex, instance.objectSlot);
    mRenderables[objectIndex] = {};
    markRenderablePoseDirty(objectIndex);
    markRenderableMetadataDirty(objectIndex);
    mRenderableIndices.erase(it);
    mDrawRegistryDirty              = true;
    mPhysicsRenderableMappingsDirty = true;
    return true;
}

bool World::removeCamera(common::EntityId entityId)
{
    const auto it = mRenderCameraIndices.find(entityId);
    if (it == mRenderCameraIndices.end())
    {
        return false;
    }

    const std::uint32_t cameraIndex    = static_cast<std::uint32_t>(it->second);
    const graphics::CameraData& camera = mRenderCameras[cameraIndex];
    reclaimDenseSlot(mFreeCameraSlotsByEnv, camera.envIndex, camera.cameraSlot);
    mRenderCameras[cameraIndex] = {};
    mRenderCameraIndices.erase(it);
    markCameraDirty(cameraIndex);
    return true;
}

bool World::removeDirectionalLight(common::EntityId entityId)
{
    const auto it = mRenderDirectionalLightIndices.find(entityId);
    if (it == mRenderDirectionalLightIndices.end())
    {
        return false;
    }

    const std::uint32_t lightIndex              = static_cast<std::uint32_t>(it->second);
    const graphics::DirectionalLightData& light = mRenderDirectionalLights[lightIndex];
    reclaimDenseSlot(mFreeDirectionalLightSlotsByEnv, light.envIndex, light.lightSlot);
    mRenderDirectionalLights[lightIndex] = {};
    mRenderDirectionalLightIndices.erase(it);
    markLightDirty(lightIndex);
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
    const auto it = mRenderableIndices.find(entityId);
    if (it == mRenderableIndices.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = static_cast<std::uint32_t>(it->second);
    MeshRendererComponent component{};
    component.mesh     = mRenderables[index].mesh;
    component.material = mRenderables[index].material;
    component.visible  = mRenderables[index].visible;
    return component;
}

std::optional<CameraComponent> World::tryGetCamera(common::EntityId entityId) const
{
    const auto it = mRenderCameraIndices.find(entityId);
    if (it == mRenderCameraIndices.end())
    {
        return std::nullopt;
    }

    const graphics::CameraData& camera = mRenderCameras[static_cast<std::uint32_t>(it->second)];
    CameraComponent component{};
    component.verticalFovDegrees = camera.verticalFovDegrees;
    component.nearClip           = camera.nearClip;
    component.farClip            = camera.farClip;
    component.outputTarget       = camera.outputTarget;
    component.outputWidth        = camera.outputWidth;
    component.outputHeight       = camera.outputHeight;
    component.viewport           = camera.viewport;
    component.clearColor         = camera.clearColor;
    component.clearDepth         = camera.clearDepth;
    component.clearColorValue    = camera.clearColorValue;
    component.clearDepthValue    = camera.clearDepthValue;
    component.renderOrder        = camera.renderOrder;
    return component;
}

std::optional<DirectionalLightComponent> World::tryGetDirectionalLight(
    common::EntityId entityId) const
{
    const auto it = mRenderDirectionalLightIndices.find(entityId);
    if (it == mRenderDirectionalLightIndices.end())
    {
        return std::nullopt;
    }

    const graphics::DirectionalLightData& light =
        mRenderDirectionalLights[static_cast<std::uint32_t>(it->second)];
    DirectionalLightComponent component{};
    component.direction          = light.direction;
    component.color              = light.color;
    component.intensity          = light.intensity;
    component.shadowDistance     = light.shadowDistance;
    component.shadowFadeDistance = light.shadowFadeDistance;
    return component;
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
    // TODO: this is not needed anymore, but we keep it for debug

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

void World::setGpuEntityScene(const gpu::GpuEntitySceneView& sceneView) noexcept
{
    mGpuEntityScene = sceneView;
}

const std::vector<graphics::RenderableInstance>& World::renderables() const noexcept
{
    return mRenderables;
}

const std::vector<graphics::CameraData>& World::cameras() const noexcept
{
    return mRenderCameras;
}

const std::vector<graphics::DirectionalLightData>& World::directionalLights() const noexcept
{
    return mRenderDirectionalLights;
}

const std::vector<Diligent::float4>& World::renderObjectPositions() const noexcept
{
    return mRenderObjectPositions;
}

const std::vector<Diligent::float4>& World::renderObjectOrientations() const noexcept
{
    return mRenderObjectOrientations;
}

const std::vector<Diligent::float4>& World::renderObjectScales() const noexcept
{
    return mRenderObjectScales;
}

const std::vector<gpu::GpuRenderableMetadata>& World::renderableMetadata() const noexcept
{
    return mRenderableMetadataHost;
}

const std::vector<gpu::GpuRenderableQueueInfo>& World::renderableQueueInfo() const noexcept
{
    return mRenderableQueueInfoHost;
}

const std::vector<gpu::GpuCameraInput>& World::cameraInputs() const noexcept
{
    return mCameraInputsHost;
}

const std::vector<gpu::GpuDirectionalLightInput>& World::lightInputs() const noexcept
{
    return mLightInputsHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry>& World::opaqueDrawRegistry()
    const noexcept
{
    return mOpaqueDrawRegistryHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry>& World::shadowDrawRegistry()
    const noexcept
{
    return mShadowDrawRegistryHost;
}

const std::vector<gpu::GpuEntityPoseMappingEntry>& World::physicsRenderableMappings()
{
    const std::uint64_t rigidBodyTopologyRevision = mPhysicsWorld.rigidBodyTopologyRevision();
    if (!mPhysicsRenderableMappingsDirty &&
        mCachedPhysicsRenderableMappingsBodyTopologyRevision == rigidBodyTopologyRevision)
    {
        return mPhysicsRenderableMappingsCache;
    }

    mPhysicsRenderableMappingsCache.clear();

    const auto& rigidBodies = mPhysicsWorld.rigidBodySoA();
    if (!rigidBodies.entityIds.empty() && !mRenderables.empty())
    {
        std::unordered_map<common::EntityId, std::uint32_t> rigidBodyIndexByEntity;
        rigidBodyIndexByEntity.reserve(rigidBodies.entityIds.size());
        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(rigidBodies.entityIds.size()); ++i)
        {
            rigidBodyIndexByEntity.emplace(rigidBodies.entityIds[i], i);
        }

        mPhysicsRenderableMappingsCache.reserve(mRenderables.size());
        for (const graphics::RenderableInstance& renderable : mRenderables)
        {
            if (renderable.entityId == common::kInvalidEntityId ||
                renderable.objectSlot == kInvalidSlot || !renderable.visible)
            {
                continue;
            }

            const auto rigidBodyIt = rigidBodyIndexByEntity.find(renderable.entityId);
            if (rigidBodyIt == rigidBodyIndexByEntity.end())
            {
                continue;
            }

            gpu::GpuEntityPoseMappingEntry entry{};
            entry.sourcePoseIndex = rigidBodyIt->second;
            entry.objectIndex =
                renderable.envIndex * mSceneLayout.maxObjectsPerEnv + renderable.objectSlot;
            mPhysicsRenderableMappingsCache.push_back(entry);
        }
    }

    mPhysicsRenderableMappingsDirty                      = false;
    mCachedPhysicsRenderableMappingsBodyTopologyRevision = rigidBodyTopologyRevision;
    return mPhysicsRenderableMappingsCache;
}

const gpu::GpuEntitySceneView& World::gpuEntityScene() const noexcept
{
    return mGpuEntityScene;
}

graphics::HostSceneView World::hostSceneView() const noexcept
{
    return graphics::HostSceneView{
        &mRenderables,
        &mRenderCameras,
        &mRenderDirectionalLights,
        &mOpaqueDrawRegistryHost,
        &mShadowDrawRegistryHost,
        &mGpuEntityScene,
    };
}

void World::ensureRenderStateUpToDate(const graphics::RenderResourceManager& resources)
{
    ensureHostSceneStorage();

    for (const std::uint32_t objectIndex : mDirtyRenderablePoseIndices)
    {
        refreshRenderablePose(objectIndex);
    }

    for (const std::uint32_t cameraIndex : mDirtyCameraIndices)
    {
        refreshCameraEntry(cameraIndex);
    }

    for (const std::uint32_t lightIndex : mDirtyLightIndices)
    {
        refreshDirectionalLightEntry(lightIndex);
    }

    refreshDirtyRenderableMetadata(resources);
    if (mDrawRegistryDirty)
    {
        rebuildDrawRegistries(resources);
        mDrawRegistryDirty = false;
    }

    clearDirtyIndexSet(mDirtyRenderablePoseIndices, mDirtyRenderablePoseBits);
    clearDirtyIndexSet(mDirtyRenderableMetadataIndices, mDirtyRenderableMetadataBits);
    clearDirtyIndexSet(mDirtyCameraIndices, mDirtyCameraBits);
    clearDirtyIndexSet(mDirtyLightIndices, mDirtyLightBits);
}

void World::refreshDirtyRenderableMetadata(const graphics::RenderResourceManager& resources)
{
    // TODO: some metadata depends on resources too, not just world state:
    // if a mesh or material changes later inside RenderResourceManager, World does
    // not currently know which object slots should become dirty.

    for (const std::uint32_t objectIndex : mDirtyRenderableMetadataIndices)
    {
        if (objectIndex >= mRenderableMetadataHost.size() || objectIndex >= mRenderables.size())
        {
            continue;
        }

        const graphics::RenderableInstance& renderable = mRenderables[objectIndex];
        gpu::GpuRenderableMetadata entry{};
        if (renderable.entityId != common::kInvalidEntityId &&
            renderable.objectSlot != kInvalidSlot)
        {
            if (renderable.visible)
            {
                entry.flags |= gpu::GpuRenderableFlag_Active;
            }

            const graphics::MaterialResourceDesc* material =
                resources.tryGetMaterial(renderable.material);
            if (material != nullptr && renderable.visible &&
                material->blendMode != graphics::BlendMode::Transparent)
            {
                entry.flags |= gpu::GpuRenderableFlag_Opaque;
                if (material->castsShadows)
                {
                    entry.flags |= gpu::GpuRenderableFlag_ShadowCaster;
                }
            }

            Diligent::float3 localBoundsMin{};
            Diligent::float3 localBoundsMax{};
            if (resources.tryGetMeshLocalBounds(renderable.mesh, localBoundsMin, localBoundsMax))
            {
                entry.localBoundsMin =
                    Diligent::float4{localBoundsMin.x, localBoundsMin.y, localBoundsMin.z, 1.0f};
                entry.localBoundsMax =
                    Diligent::float4{localBoundsMax.x, localBoundsMax.y, localBoundsMax.z, 1.0f};
            }
        }

        mRenderableMetadataHost[objectIndex] = entry;
    }
}

void World::rebuildDrawRegistries(const graphics::RenderResourceManager& resources)
{
    // TODO: mesh/material resources still authored on CPU.
    // I don't know if we can let GPU do this part completely.

    std::map<DrawBucketKey, std::vector<std::uint32_t>> opaqueObjectsByKey;
    std::map<DrawBucketKey, std::vector<std::uint32_t>> shadowObjectsByKey;
    mRenderableQueueInfoHost.assign(mRenderables.size(), gpu::GpuRenderableQueueInfo{});
    mOpaqueDrawRegistryHost.clear();
    mShadowDrawRegistryHost.clear();

    for (std::uint32_t objectIndex = 0u;
         objectIndex < static_cast<std::uint32_t>(mRenderables.size()); ++objectIndex)
    {
        const graphics::RenderableInstance& renderable = mRenderables[objectIndex];
        if (renderable.entityId == common::kInvalidEntityId ||
            renderable.objectSlot == kInvalidSlot || !renderable.visible)
        {
            continue;
        }

        const graphics::MeshResourceDesc* mesh = resources.tryGetMesh(renderable.mesh);
        const graphics::MaterialResourceDesc* material =
            resources.tryGetMaterial(renderable.material);
        if (mesh == nullptr || material == nullptr ||
            material->blendMode == graphics::BlendMode::Transparent || mesh->vertices.empty() ||
            mesh->indices.size() < 3)
        {
            continue;
        }

        const DrawBucketKey key{
            material->pipeline.programFamily,
            static_cast<std::uint32_t>(material->pipeline.featureFlags),
            renderable.material.id,
            renderable.mesh.id,
        };
        opaqueObjectsByKey[key].push_back(objectIndex);
        if (material->castsShadows)
        {
            shadowObjectsByKey[key].push_back(objectIndex);
        }
    }

    std::uint32_t opaqueCommandIndex = 0u;
    for (const auto& [key, objectIndices] : opaqueObjectsByKey)
    {
        if (objectIndices.empty())
        {
            continue;
        }

        const graphics::MeshResourceDesc* mesh =
            resources.tryGetMesh(graphics::MeshHandle{key.meshId});
        if (mesh == nullptr)
        {
            continue;
        }

        graphics::IndirectCommandRegistryEntry entry{};
        entry.drawCommand.useDrawListBuffer = 1u;
        entry.drawCommand.programFamily     = key.programFamily;
        entry.drawCommand.materialFeatureFlags =
            static_cast<graphics::MaterialFeatureFlags>(key.materialFeatureFlags);
        entry.drawCommand.meshId      = key.meshId;
        entry.drawCommand.materialId  = key.materialId;
        entry.drawCommand.meshVersion = resources.meshVersion(graphics::MeshHandle{key.meshId});
        entry.drawCommand.indexCount  = static_cast<std::uint32_t>(mesh->indices.size());
        entry.maxVisibleCount         = static_cast<std::uint32_t>(objectIndices.size());
        mOpaqueDrawRegistryHost.push_back(entry);
        for (const std::uint32_t objectIndex : objectIndices)
        {
            mRenderableQueueInfoHost[objectIndex].opaqueCommandIndex = opaqueCommandIndex;
        }
        ++opaqueCommandIndex;
    }

    std::uint32_t shadowCommandBaseIndex = 0u;
    for (const auto& [key, objectIndices] : shadowObjectsByKey)
    {
        if (objectIndices.empty())
        {
            continue;
        }

        const graphics::MeshResourceDesc* mesh =
            resources.tryGetMesh(graphics::MeshHandle{key.meshId});
        if (mesh == nullptr)
        {
            continue;
        }

        for (std::uint32_t cascadeIndex = 0u; cascadeIndex < graphics::kShadowCascadeCount;
             ++cascadeIndex)
        {
            graphics::IndirectCommandRegistryEntry entry{};
            entry.drawCommand.useDrawListBuffer = 1u;
            entry.drawCommand.programFamily     = key.programFamily;
            entry.drawCommand.materialFeatureFlags =
                static_cast<graphics::MaterialFeatureFlags>(key.materialFeatureFlags);
            entry.drawCommand.meshId      = key.meshId;
            entry.drawCommand.materialId  = key.materialId;
            entry.drawCommand.meshVersion = resources.meshVersion(graphics::MeshHandle{key.meshId});
            entry.drawCommand.indexCount  = static_cast<std::uint32_t>(mesh->indices.size());
            entry.maxVisibleCount         = static_cast<std::uint32_t>(objectIndices.size());
            mShadowDrawRegistryHost.push_back(entry);
        }

        for (const std::uint32_t objectIndex : objectIndices)
        {
            mRenderableQueueInfoHost[objectIndex].shadowCommandBaseIndex = shadowCommandBaseIndex;
        }
        shadowCommandBaseIndex += graphics::kShadowCascadeCount;
    }
}

void World::refreshRenderablePose(std::uint32_t objectIndex)
{
    if (objectIndex >= mRenderables.size())
    {
        return;
    }

    graphics::RenderableInstance& renderable = mRenderables[objectIndex];
    if (renderable.entityId == common::kInvalidEntityId || renderable.objectSlot == kInvalidSlot)
    {
        mRenderObjectPositions[objectIndex]    = Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f};
        mRenderObjectOrientations[objectIndex] = Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f};
        mRenderObjectScales[objectIndex]       = Diligent::float4{1.0f, 1.0f, 1.0f, 0.0f};
        return;
    }

    renderable.worldTransform =
        tryGetTransform(renderable.entityId).value_or(TransformComponent{}).worldTransform;
    mRenderObjectPositions[objectIndex] =
        Diligent::float4{renderable.worldTransform.position.x, renderable.worldTransform.position.y,
                         renderable.worldTransform.position.z, 1.0f};
    mRenderObjectOrientations[objectIndex] = Diligent::float4{
        renderable.worldTransform.rotation.q.x, renderable.worldTransform.rotation.q.y,
        renderable.worldTransform.rotation.q.z, renderable.worldTransform.rotation.q.w};
    mRenderObjectScales[objectIndex] =
        Diligent::float4{renderable.worldTransform.scale.x, renderable.worldTransform.scale.y,
                         renderable.worldTransform.scale.z, 0.0f};
}

void World::refreshCameraEntry(std::uint32_t cameraIndex)
{
    if (cameraIndex >= mRenderCameras.size())
    {
        return;
    }

    graphics::CameraData& cameraData = mRenderCameras[cameraIndex];
    if (cameraData.entityId == common::kInvalidEntityId || cameraData.cameraSlot == kInvalidSlot)
    {
        mCameraInputsHost[cameraIndex] = {};
        return;
    }

    cameraData.worldTransform =
        tryGetTransform(cameraData.entityId).value_or(TransformComponent{}).worldTransform;

    gpu::GpuCameraInput input{};
    input.position =
        Diligent::float4{cameraData.worldTransform.position.x, cameraData.worldTransform.position.y,
                         cameraData.worldTransform.position.z, 1.0f};
    input.orientation = Diligent::float4{
        cameraData.worldTransform.rotation.q.x, cameraData.worldTransform.rotation.q.y,
        cameraData.worldTransform.rotation.q.z, cameraData.worldTransform.rotation.q.w};
    input.projectionParams = Diligent::float4{cameraData.verticalFovDegrees, cameraData.nearClip,
                                              cameraData.farClip, 0.0f};
    input.viewportAndOutputSize = Diligent::float4{
        cameraData.viewport.width, cameraData.viewport.height,
        static_cast<float>(cameraData.outputWidth), static_cast<float>(cameraData.outputHeight)};
    input.envIndex                 = cameraData.envIndex;
    input.cameraSlot               = cameraData.cameraSlot;
    input.active                   = 1u;
    mCameraInputsHost[cameraIndex] = input;
}

void World::refreshDirectionalLightEntry(std::uint32_t lightIndex)
{
    if (lightIndex >= mRenderDirectionalLights.size())
    {
        return;
    }

    const graphics::DirectionalLightData& lightData = mRenderDirectionalLights[lightIndex];
    if (lightData.entityId == common::kInvalidEntityId || lightData.lightSlot == kInvalidSlot)
    {
        mLightInputsHost[lightIndex] = {};
        return;
    }

    gpu::GpuDirectionalLightInput input{};
    input.directionIntensity = Diligent::float4{lightData.direction.x, lightData.direction.y,
                                                lightData.direction.z, lightData.intensity};
    input.color = Diligent::float4{lightData.color.x, lightData.color.y, lightData.color.z, 0.0f};
    input.shadowParams =
        Diligent::float4{lightData.shadowDistance, lightData.shadowFadeDistance, 0.0f, 0.0f};
    input.envIndex               = lightData.envIndex;
    input.lightSlot              = lightData.lightSlot;
    input.active                 = 1u;
    mLightInputsHost[lightIndex] = input;
}

void World::moveRenderableToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    const auto indexIt = mRenderableIndices.find(entityId);
    if (indexIt == mRenderableIndices.end())
    {
        return;
    }

    const std::uint32_t oldObjectIndex      = static_cast<std::uint32_t>(indexIt->second);
    graphics::RenderableInstance renderable = mRenderables[oldObjectIndex];
    reclaimDenseSlot(mFreeRenderableSlotsByEnv, renderable.envIndex, renderable.objectSlot);
    mRenderables[oldObjectIndex] = {};
    renderable.envIndex          = envIndex;
    renderable.objectSlot =
        allocateDenseSlot(mFreeRenderableSlotsByEnv, mNextRenderableSlotByEnv, envIndex,
                          mSceneLayout.maxObjectsPerEnv, "renderable");
    const std::uint32_t newObjectIndex =
        envIndex * mSceneLayout.maxObjectsPerEnv + renderable.objectSlot;
    mRenderables[newObjectIndex] = renderable;
    markRenderableMetadataDirty(oldObjectIndex);
    markRenderableMetadataDirty(newObjectIndex);
    markRenderablePoseDirty(oldObjectIndex);
    markRenderablePoseDirty(newObjectIndex);
    indexIt->second = newObjectIndex;
}

void World::moveCameraToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    const auto indexIt = mRenderCameraIndices.find(entityId);
    if (indexIt == mRenderCameraIndices.end())
    {
        return;
    }

    const std::uint32_t oldCameraIndex = static_cast<std::uint32_t>(indexIt->second);
    graphics::CameraData camera        = mRenderCameras[oldCameraIndex];
    reclaimDenseSlot(mFreeCameraSlotsByEnv, camera.envIndex, camera.cameraSlot);
    mRenderCameras[oldCameraIndex] = {};
    camera.envIndex                = envIndex;
    camera.cameraSlot = allocateDenseSlot(mFreeCameraSlotsByEnv, mNextCameraSlotByEnv, envIndex,
                                          mSceneLayout.maxCamerasPerEnv, "camera");
    const std::uint32_t newCameraIndex =
        envIndex * mSceneLayout.maxCamerasPerEnv + camera.cameraSlot;
    mRenderCameras[newCameraIndex] = camera;
    indexIt->second                = newCameraIndex;
    markCameraDirty(oldCameraIndex);
    markCameraDirty(newCameraIndex);
}

void World::moveDirectionalLightToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    const auto indexIt = mRenderDirectionalLightIndices.find(entityId);
    if (indexIt == mRenderDirectionalLightIndices.end())
    {
        return;
    }

    const std::uint32_t oldLightIndex    = static_cast<std::uint32_t>(indexIt->second);
    graphics::DirectionalLightData light = mRenderDirectionalLights[oldLightIndex];
    reclaimDenseSlot(mFreeDirectionalLightSlotsByEnv, light.envIndex, light.lightSlot);
    mRenderDirectionalLights[oldLightIndex] = {};
    light.envIndex                          = envIndex;
    light.lightSlot =
        allocateDenseSlot(mFreeDirectionalLightSlotsByEnv, mNextDirectionalLightSlotByEnv, envIndex,
                          mSceneLayout.maxLightsPerEnv, "directional light");
    const std::uint32_t newLightIndex = envIndex * mSceneLayout.maxLightsPerEnv + light.lightSlot;
    mRenderDirectionalLights[newLightIndex] = light;
    indexIt->second                         = newLightIndex;
    markLightDirty(oldLightIndex);
    markLightDirty(newLightIndex);
}

} // namespace cressim::neo::engine
