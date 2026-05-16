#include "engine/world.h"
#include "common/logger.h"

#include <array>
#include <cmath>
#include <map>

namespace cressim::neo::engine
{

namespace
{
constexpr std::uint32_t kInvalidSlot        = 0xffffffffu;
constexpr float kSoftBodyVertexMatchEpsilon = 1.0e-3f;

bool rendersInTransparentPass(const graphics::MaterialResourceDesc &material) noexcept
{
    return graphics::usesTransparentPass(material);
}

struct DrawBucketKey
{
    std::int32_t renderOrder                      = 0;
    graphics::MaterialProgramFamily programFamily = graphics::MaterialProgramFamily::StandardLit;
    std::uint32_t materialFeatureFlags            = 0u;
    common::ResourceId materialId                 = common::kInvalidResourceId;
    common::ResourceId meshId                     = common::kInvalidResourceId;

    [[nodiscard]] bool operator<(const DrawBucketKey &rhs) const noexcept
    {
        if (renderOrder != rhs.renderOrder)
        {
            return renderOrder < rhs.renderOrder;
        }
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

const std::vector<World::ColliderHandle> &emptyColliderHandleList()
{
    static const std::vector<World::ColliderHandle> kEmpty;
    return kEmpty;
}

std::uint32_t allocateDenseSlot(
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> &freeSlotsByEnv,
    std::unordered_map<std::uint32_t, std::uint32_t> &nextSlotByEnv, std::uint32_t envIndex,
    std::uint32_t maxSlotsPerEnv, const char *slotType)
{
    auto &freeSlots = freeSlotsByEnv[envIndex];
    if (!freeSlots.empty())
    {
        const std::uint32_t slot = freeSlots.back();
        freeSlots.pop_back();
        return slot;
    }

    std::uint32_t &nextSlot = nextSlotByEnv[envIndex];
    if (nextSlot >= maxSlotsPerEnv)
    {
        CRESSIM_LOG_ERROR(slotType, " capacity exceeded for environment ", envIndex, ".");
        return kInvalidSlot;
    }
    return nextSlot++;
}

void reclaimDenseSlot(std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> &freeSlotsByEnv,
                      std::uint32_t envIndex, std::uint32_t slot)
{
    if (slot == kInvalidSlot)
    {
        return;
    }
    freeSlotsByEnv[envIndex].push_back(slot);
}

void enqueueDenseDirtyIndex(std::uint32_t index, std::vector<std::uint32_t> &dirtyIndices,
                            std::vector<std::uint8_t> &dirtyBits)
{
    if (index >= dirtyBits.size() || dirtyBits[index] != 0u)
    {
        return;
    }

    dirtyBits[index] = 1u;
    dirtyIndices.push_back(index);
}

bool isValidLight(const graphics::LightData &light) noexcept
{
    return light.entityId != common::kInvalidEntityId && light.lightSlot != kInvalidSlot;
}

void refreshTransformDerivedLightState(graphics::LightData &light,
                                       const TransformComponent &transform) noexcept
{
    switch (light.type)
    {
    case graphics::GpuLightType::Directional:
        // Directional lights currently use authored direction, but keeping position refreshed
        // makes the unified host cache consistent across all light types.
        light.position = transform.worldTransform.position;
        break;
    case graphics::GpuLightType::Point:
        light.position = transform.worldTransform.position;
        break;
    case graphics::GpuLightType::Spot:
        light.position = transform.worldTransform.position;
        break;
    }
}

struct QuantizedPointKey
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    [[nodiscard]] bool operator==(const QuantizedPointKey &rhs) const noexcept
    {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct QuantizedPointKeyHash
{
    [[nodiscard]] std::size_t operator()(const QuantizedPointKey &key) const noexcept
    {
        const std::size_t hx = std::hash<std::int64_t>{}(key.x);
        const std::size_t hy = std::hash<std::int64_t>{}(key.y);
        const std::size_t hz = std::hash<std::int64_t>{}(key.z);
        return hx ^ (hy << 1u) ^ (hz << 2u);
    }
};

QuantizedPointKey quantizePoint(const Diligent::float3 &point, float epsilon) noexcept
{
    const double invEpsilon = 1.0 / static_cast<double>(std::max(epsilon, 1.0e-8f));
    return QuantizedPointKey{
        static_cast<std::int64_t>(std::llround(static_cast<double>(point.x) * invEpsilon)),
        static_cast<std::int64_t>(std::llround(static_cast<double>(point.y) * invEpsilon)),
        static_cast<std::int64_t>(std::llround(static_cast<double>(point.z) * invEpsilon)),
    };
}

Diligent::float3 transformPoint(const common::Transform &transform,
                                const Diligent::float3 &localPoint) noexcept
{
    const Diligent::float3 scaled{localPoint.x * transform.scale.x,
                                  localPoint.y * transform.scale.y,
                                  localPoint.z * transform.scale.z};
    return transform.rotation.RotateVector(scaled) + transform.position;
}

Diligent::float3 inverseTransformPoint(const common::Transform &transform,
                                       const Diligent::float3 &worldPoint) noexcept
{
    const Diligent::QuaternionF inverseRotation{-transform.rotation.q.x, -transform.rotation.q.y,
                                                -transform.rotation.q.z, transform.rotation.q.w};
    const Diligent::float3 localScaled =
        inverseRotation.RotateVector(worldPoint - transform.position);
    const Diligent::float3 safeScale{
        std::max(std::abs(transform.scale.x), 1.0e-6f),
        std::max(std::abs(transform.scale.y), 1.0e-6f),
        std::max(std::abs(transform.scale.z), 1.0e-6f),
    };
    return {localScaled.x / safeScale.x, localScaled.y / safeScale.y, localScaled.z / safeScale.z};
}

Diligent::float3 transformNormal(const common::Transform &transform,
                                 const Diligent::float3 &localNormal) noexcept
{
    const Diligent::float3 safeScale{
        std::max(std::abs(transform.scale.x), 1.0e-6f),
        std::max(std::abs(transform.scale.y), 1.0e-6f),
        std::max(std::abs(transform.scale.z), 1.0e-6f),
    };
    Diligent::float3 worldNormal = transform.rotation.RotateVector(Diligent::float3{
        localNormal.x / safeScale.x, localNormal.y / safeScale.y, localNormal.z / safeScale.z});
    const float lenSq            = Diligent::dot(worldNormal, worldNormal);
    if (lenSq <= 1.0e-12f)
    {
        return {0.0f, 1.0f, 0.0f};
    }
    return worldNormal * (1.0f / std::sqrt(lenSq));
}

std::optional<std::uint32_t> findMatchingRestVertexLocal(
    const Diligent::float3 &visualVertexLocal,
    const std::vector<Diligent::float3> &restPositionsLocal,
    const std::unordered_map<QuantizedPointKey, std::uint32_t, QuantizedPointKeyHash>
        &restIndexByQuantizedPosition,
    float epsilon) noexcept
{
    const QuantizedPointKey key = quantizePoint(visualVertexLocal, epsilon);
    if (const auto it = restIndexByQuantizedPosition.find(key);
        it != restIndexByQuantizedPosition.end())
    {
        return it->second;
    }

    const float epsilonSq = epsilon * epsilon;
    float bestDistanceSq  = epsilonSq;
    std::optional<std::uint32_t> bestIndex;
    for (std::uint32_t localParticleIndex = 0u;
         localParticleIndex < static_cast<std::uint32_t>(restPositionsLocal.size());
         ++localParticleIndex)
    {
        const Diligent::float3 delta = restPositionsLocal[localParticleIndex] - visualVertexLocal;
        const float distanceSq       = Diligent::dot(delta, delta);
        if (distanceSq > bestDistanceSq)
        {
            continue;
        }

        bestDistanceSq = distanceSq;
        bestIndex      = localParticleIndex;
    }

    return bestIndex;
}

} // namespace

common::EntityId World::createEntity(std::uint32_t envIndex)
{
    // A globally unique EntityId within this World.

    if (envIndex >= mSceneLayout.envCount)
    {
        CRESSIM_LOG_ERROR("createEntity envIndex exceeds configured environment count.");
        return common::kInvalidEntityId;
    }
    ensureHostSceneStorage();
    mWorldSceneAuthored = true;

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
    removePointLight(entityId);
    removeSpotLight(entityId);
    removeRigidBody(entityId);
    removeSoftBody(entityId);
    removeFluid(entityId);

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

void World::setSceneLayout(const common::SceneLayoutDesc &layout)
{
    if (mWorldSceneAuthored)
    {
        CRESSIM_LOG_ERROR("cannot re-configure scene layout after authoring.");
        return;
    }

    mSceneLayout = layout;
    ensureHostSceneStorage();
}

const common::SceneLayoutDesc &World::sceneLayout() const noexcept
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
        CRESSIM_LOG_ERROR("setEntityEnvironment envIndex exceeds configured environment count.");
        return false;
    }

    const std::uint32_t previousEnv = entityEnvironment(entityId);
    if (previousEnv == envIndex)
    {
        return true;
    }

    const auto physIt = mPhysicsLinks.find(entityId);
    if (physIt != mPhysicsLinks.end())
    {
        if (physIt->second.hasRigidBody)
        {
            if (physics::RigidBodyState *rigidBody = mPhysicsWorld.tryGetRigidBody(entityId))
            {
                physics::RigidBodyState updated = *rigidBody;
                updated.environmentIndex        = envIndex;
                mPhysicsWorld.upsertRigidBody(updated);
            }
        }
        if (physIt->second.hasSoftBody)
        {
            if (physics::SoftBodyState *softBody = mPhysicsWorld.tryGetSoftBody(entityId))
            {
                physics::SoftBodyState updated = *softBody;
                updated.environmentIndex       = envIndex;
                if (!mPhysicsWorld.upsertSoftBody(updated))
                {
                    return false;
                }
            }
        }
        if (physIt->second.hasFluid)
        {
            if (physics::FluidState *fluid = mPhysicsWorld.tryGetFluid(entityId))
            {
                physics::FluidState updated = *fluid;
                updated.environmentIndex    = envIndex;
                if (!mPhysicsWorld.upsertFluid(updated))
                {
                    return false;
                }
            }
        }
    }
    mEntityEnvironments[entityId] = envIndex;
    if (!moveRenderableToEnvironment(entityId, envIndex) ||
        !moveCameraToEnvironment(entityId, envIndex) || !moveLightToEnvironment(entityId, envIndex))
    {
        mEntityEnvironments[entityId] = previousEnv;
        if (physIt != mPhysicsLinks.end())
        {
            if (physIt->second.hasRigidBody)
            {
                if (physics::RigidBodyState *rigidBody = mPhysicsWorld.tryGetRigidBody(entityId))
                {
                    physics::RigidBodyState reverted = *rigidBody;
                    reverted.environmentIndex        = previousEnv;
                    mPhysicsWorld.upsertRigidBody(reverted);
                }
            }
            if (physIt->second.hasSoftBody)
            {
                if (physics::SoftBodyState *softBody = mPhysicsWorld.tryGetSoftBody(entityId))
                {
                    physics::SoftBodyState reverted = *softBody;
                    reverted.environmentIndex       = previousEnv;
                    if (!mPhysicsWorld.upsertSoftBody(reverted))
                    {
                        CRESSIM_LOG_ERROR("setEntityEnvironment failed to restore previous "
                                          "soft-body environment for entity ",
                                          entityId, ".");
                    }
                }
            }
            if (physIt->second.hasFluid)
            {
                if (physics::FluidState *fluid = mPhysicsWorld.tryGetFluid(entityId))
                {
                    physics::FluidState reverted = *fluid;
                    reverted.environmentIndex    = previousEnv;
                    if (!mPhysicsWorld.upsertFluid(reverted))
                    {
                        CRESSIM_LOG_ERROR("setEntityEnvironment failed to restore previous fluid "
                                          "environment for entity ",
                                          entityId, ".");
                    }
                }
            }
        }
        return false;
    }
    mDrawRegistryDirty              = true;
    mPhysicsRenderableMappingsDirty = true;
    return true;
}

std::uint32_t World::entityEnvironment(common::EntityId entityId) const noexcept
{
    const auto it = mEntityEnvironments.find(entityId);
    return it != mEntityEnvironments.end() ? it->second : 0u;
}

bool World::setEnvironmentIbl(std::uint32_t envIndex, const graphics::EnvironmentIblDesc &desc)
{
    if (envIndex >= mSceneLayout.envCount)
    {
        return false;
    }

    // In case user did not explicitly set the layout.
    ensureHostSceneStorage();

    // Setting authoring status to true prevents scene layout change after this.
    mWorldSceneAuthored = true;

    mEnvironmentIbls[envIndex] = desc;
    return true;
}

const graphics::EnvironmentIblDesc *World::tryGetEnvironmentIbl(
    std::uint32_t envIndex) const noexcept
{
    if (envIndex >= mEnvironmentIbls.size())
    {
        return nullptr;
    }
    return &mEnvironmentIbls[envIndex];
}

bool World::setEnvironmentFluid(std::uint32_t envIndex,
                                const graphics::EnvironmentFluidDesc &desc)
{
    if (envIndex >= mSceneLayout.envCount)
    {
        return false;
    }

    ensureHostSceneStorage();
    mWorldSceneAuthored             = true;
    mEnvironmentFluids[envIndex] = desc;
    return true;
}

const graphics::EnvironmentFluidDesc *World::tryGetEnvironmentFluid(
    std::uint32_t envIndex) const noexcept
{
    if (envIndex >= mEnvironmentFluids.size())
    {
        return nullptr;
    }
    return &mEnvironmentFluids[envIndex];
}

bool World::isAlive(common::EntityId entityId) const
{
    return mAlive.find(entityId) != mAlive.end();
}

const std::vector<common::EntityId> &World::entities() const noexcept
{
    return mEntities;
}

bool World::requireAliveEntity(common::EntityId entityId, const char *operation) const noexcept
{
    if (mAlive.find(entityId) != mAlive.end())
    {
        return true;
    }

    CRESSIM_LOG_ERROR(operation, " requires an existing entity id (entityId=", entityId, ").");
    return false;
}

void World::ensureHostSceneStorage()
{
    const std::size_t objectCapacity = mSceneLayout.totalRenderableObjectCapacity();
    if (mRenderables.size() != objectCapacity)
    {
        mRenderables.assign(objectCapacity, graphics::RenderableInstance{});
        mRenderObjectPositions.assign(objectCapacity, Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f});
        mRenderObjectOrientations.assign(objectCapacity, Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f});
        mRenderObjectScales.assign(objectCapacity, Diligent::float4{1.0f, 1.0f, 1.0f, 0.0f});
        mRenderableMetadataHost.assign(objectCapacity, graphics::GpuRenderableMetadata{});
        mRenderableQueueInfoHost.assign(objectCapacity, graphics::GpuRenderableQueueInfo{});
        mSoftBodyVertexBindingBaseByObject.assign(objectCapacity, kInvalidSlot);
        mSoftBodyVertexNormalBaseByObject.assign(objectCapacity, kInvalidSlot);
        mSoftBodyVertexCountByObject.assign(objectCapacity, 0u);
        mOpaqueDrawRegistryHost.clear();
        mTransparentDrawRegistryHost.clear();
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
        mSoftBodyRenderBindingsDirty    = true;
    }

    const std::size_t cameraCapacity = mSceneLayout.totalCameraCapacity();
    if (mRenderCameras.size() != cameraCapacity)
    {
        mRenderCameras.assign(cameraCapacity, graphics::CameraData{});
        mCameraInputsHost.assign(cameraCapacity, graphics::GpuCameraInput{});
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
    if (mRenderLights.size() != lightCapacity)
    {
        mRenderLights.assign(lightCapacity, graphics::LightData{});
        mLightInputsHost.assign(lightCapacity, graphics::GpuLightInput{});
        mLocalLightSelectionsHost.assign(mSceneLayout.envCount, graphics::GpuLocalLightSelection{});
        mDirtyLightIndices.clear();
        mDirtyLightBits.assign(lightCapacity, 0u);
        mDirtyLightIndices.reserve(lightCapacity);
        for (std::uint32_t i = 0; i < lightCapacity; ++i)
        {
            mDirtyLightIndices.push_back(i);
            mDirtyLightBits[i] = 1u;
        }
    }

    if (mEnvironmentIbls.size() != mSceneLayout.envCount)
    {
        mEnvironmentIbls.resize(mSceneLayout.envCount, graphics::EnvironmentIblDesc{});
    }
    if (mEnvironmentFluids.size() != mSceneLayout.envCount)
    {
        mEnvironmentFluids.resize(mSceneLayout.envCount, graphics::EnvironmentFluidDesc{});
    }
    if (mLocalLightSelectionsHost.size() != mSceneLayout.envCount)
    {
        mLocalLightSelectionsHost.assign(mSceneLayout.envCount, graphics::GpuLocalLightSelection{});
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

void World::clearDirtyIndexSet(std::vector<std::uint32_t> &dirtyIndices,
                               std::vector<std::uint8_t> &dirtyBits)
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

void World::setTransform(common::EntityId entityId, const TransformComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setTransform requires valid entity id.");
        return;
    }

    if (!requireAliveEntity(entityId, "setTransform"))
    {
        return;
    }

    const auto it = mTransformIndex.find(entityId);
    if (it == mTransformIndex.end())
    {
        const std::uint32_t newIndex = static_cast<std::uint32_t>(mTransforms.entityIds.size());
        mTransforms.entityIds.push_back(entityId);
        mTransforms.components.push_back(component);
        mTransformIndex.emplace(entityId, newIndex);
    }
    else
    {
        mTransforms.components[it->second] = component;
    }

    // Forcefully set rigid body transform from world transform. This teleports the body (no physics
    // integration). Users should not set the transform direction on an entity with a kinematic
    // rigid body.
    if (auto *rb = mPhysicsWorld.tryGetRigidBody(entityId))
    {
        rb->position = component.worldTransform.position;
        rb->rotation = component.worldTransform.rotation;
        rb->scale    = component.worldTransform.scale;
        mPhysicsWorld.upsertRigidBody(*rb);
    }
    if (auto *softBody = mPhysicsWorld.tryGetSoftBody(entityId))
    {
        physics::SoftBodyState updated = *softBody;
        updated.restTransform          = component.worldTransform;
        if (!mPhysicsWorld.upsertSoftBody(updated))
        {
            CRESSIM_LOG_ERROR("setTransform failed to rebuild soft body for entity ", entityId,
                              ".");
        }
    }
    if (auto *fluid = mPhysicsWorld.tryGetFluid(entityId))
    {
        physics::FluidState updated = *fluid;
        updated.restTransform       = component.worldTransform;
        if (!mPhysicsWorld.upsertFluid(updated))
        {
            CRESSIM_LOG_ERROR("setTransform failed to rebuild fluid for entity ", entityId, ".");
        }
    }
    if (const auto it = mRenderableIndices.find(entityId); it != mRenderableIndices.end())
    {
        markRenderablePoseDirty(static_cast<std::uint32_t>(it->second));
    }
    if (const auto it = mRenderCameraIndices.find(entityId); it != mRenderCameraIndices.end())
    {
        markCameraDirty(static_cast<std::uint32_t>(it->second));
    }
    if (const auto it = mRenderLightIndices.find(entityId); it != mRenderLightIndices.end())
    {
        markLightDirty(static_cast<std::uint32_t>(it->second));
    }
}

void World::setMeshRenderer(common::EntityId entityId, const MeshRendererComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setMeshRenderer requires valid entity id.");
        return;
    }

    if (!requireAliveEntity(entityId, "setMeshRenderer"))
    {
        return;
    }

    const auto indexIt        = mRenderableIndices.find(entityId);
    std::uint32_t objectIndex = 0u;
    if (indexIt == mRenderableIndices.end())
    {
        const std::uint32_t envIndex = entityEnvironment(entityId);
        const std::uint32_t objectSlot =
            allocateDenseSlot(mFreeRenderableSlotsByEnv, mNextRenderableSlotByEnv, envIndex,
                              mSceneLayout.maxRenderableObjectsPerEnv, "renderable");
        if (objectSlot == kInvalidSlot)
        {
            return;
        }
        objectIndex = envIndex * mSceneLayout.maxRenderableObjectsPerEnv + objectSlot;
        mRenderableIndices[entityId]         = objectIndex;
        mRenderables[objectIndex].entityId   = entityId;
        mRenderables[objectIndex].envIndex   = envIndex;
        mRenderables[objectIndex].objectSlot = objectSlot;
    }
    else
    {
        objectIndex = static_cast<std::uint32_t>(indexIt->second);
    }

    graphics::RenderableInstance &renderable = mRenderables[objectIndex];
    renderable.mesh                          = component.mesh;
    renderable.material                      = component.material;
    renderable.visible                       = component.visible;
    markRenderableMetadataDirty(objectIndex);
    markRenderablePoseDirty(objectIndex);
    mDrawRegistryDirty              = true;
    mPhysicsRenderableMappingsDirty = true;
    mSoftBodyRenderBindingsDirty    = true;
}

void World::setCamera(common::EntityId entityId, const CameraComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setCamera requires valid entity id.");
        return;
    }

    if (!requireAliveEntity(entityId, "setCamera"))
    {
        return;
    }

    const auto indexIt        = mRenderCameraIndices.find(entityId);
    std::uint32_t cameraIndex = 0u;
    if (indexIt == mRenderCameraIndices.end())
    {
        const std::uint32_t envIndex = entityEnvironment(entityId);
        const std::uint32_t cameraSlot =
            allocateDenseSlot(mFreeCameraSlotsByEnv, mNextCameraSlotByEnv, envIndex,
                              mSceneLayout.maxCamerasPerEnv, "camera");
        if (cameraSlot == kInvalidSlot)
        {
            return;
        }
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

    graphics::CameraData &cameraData = mRenderCameras[cameraIndex];
    cameraData.worldTransform =
        tryGetTransform(entityId).value_or(TransformComponent{}).worldTransform;
    cameraData.verticalFovDegrees = component.verticalFovDegrees;
    cameraData.nearClip           = component.nearClip;
    cameraData.farClip            = component.farClip;
    cameraData.output             = component.output;
    cameraData.outputWidth        = component.outputWidth;
    cameraData.outputHeight       = component.outputHeight;
    cameraData.viewport           = component.viewport;
    cameraData.clearColor         = component.clearColor;
    cameraData.clearDepth         = component.clearDepth;
    cameraData.clearColorValue    = component.clearColorValue;
    cameraData.clearDepthValue    = component.clearDepthValue;
    cameraData.backgroundMode =
        component.backgroundMode == CameraComponent::BackgroundMode::EnvironmentCubemap
            ? graphics::CameraBackgroundMode::EnvironmentCubemap
            : graphics::CameraBackgroundMode::ClearColor;
    cameraData.renderOrder = component.renderOrder;
    markCameraDirty(cameraIndex);
}

void World::setDirectionalLight(common::EntityId entityId,
                                const DirectionalLightComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setDirectionalLight requires valid entity id.");
        return;
    }

    if (!requireAliveEntity(entityId, "setDirectionalLight"))
    {
        return;
    }

    std::uint32_t lightIndex = 0u;
    if (!tryGetLightIndexForType(entityId, graphics::GpuLightType::Directional,
                                 "setDirectionalLight", lightIndex))
    {
        return;
    }

    if (lightIndex == kInvalidSlot)
    {
        const std::uint32_t envIndex  = entityEnvironment(entityId);
        const std::uint32_t lightSlot = allocateLightSlot(envIndex, true);
        if (lightSlot == kInvalidSlot)
        {
            return;
        }
        lightIndex                          = envIndex * mSceneLayout.maxLightsPerEnv + lightSlot;
        mRenderLightIndices[entityId]       = lightIndex;
        mRenderLights[lightIndex].entityId  = entityId;
        mRenderLights[lightIndex].envIndex  = envIndex;
        mRenderLights[lightIndex].lightSlot = lightSlot;
    }

    graphics::LightData &lightData     = mRenderLights[lightIndex];
    const TransformComponent transform = tryGetTransform(entityId).value_or(TransformComponent{});
    lightData.type                     = graphics::GpuLightType::Directional;
    lightData.position                 = transform.worldTransform.position;
    lightData.direction                = component.direction;
    lightData.color                    = component.color;
    lightData.intensity                = component.intensity;
    lightData.range                    = component.range;
    lightData.innerConeAngle           = 0.0f;
    lightData.outerConeAngle           = 0.0f;
    lightData.shadowDistance           = component.shadowDistance;
    lightData.shadowFadeDistance       = component.shadowFadeDistance;
    lightData.shadowBias               = component.shadowBias;
    lightData.castsShadows             = component.castsShadows;
    markLightDirty(lightIndex);
}

void World::setPointLight(common::EntityId entityId, const PointLightComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setPointLight requires valid entity id.");
        return;
    }

    if (!requireAliveEntity(entityId, "setPointLight"))
    {
        return;
    }

    std::uint32_t lightIndex = 0u;
    if (!tryGetLightIndexForType(entityId, graphics::GpuLightType::Point, "setPointLight",
                                 lightIndex))
    {
        return;
    }

    if (lightIndex == kInvalidSlot)
    {
        const std::uint32_t envIndex  = entityEnvironment(entityId);
        const std::uint32_t lightSlot = allocateLightSlot(envIndex, false);
        if (lightSlot == kInvalidSlot)
        {
            return;
        }
        lightIndex                          = envIndex * mSceneLayout.maxLightsPerEnv + lightSlot;
        mRenderLightIndices[entityId]       = lightIndex;
        mRenderLights[lightIndex].entityId  = entityId;
        mRenderLights[lightIndex].envIndex  = envIndex;
        mRenderLights[lightIndex].lightSlot = lightSlot;
    }

    const TransformComponent transform = tryGetTransform(entityId).value_or(TransformComponent{});
    graphics::LightData &lightData     = mRenderLights[lightIndex];
    lightData.type                     = graphics::GpuLightType::Point;
    lightData.position                 = transform.worldTransform.position;
    lightData.direction                = Diligent::float3{0.0f, -1.0f, 0.0f};
    lightData.color                    = component.color;
    lightData.intensity                = component.intensity;
    lightData.range                    = component.range;
    lightData.innerConeAngle           = 0.0f;
    lightData.outerConeAngle           = 0.0f;
    lightData.shadowDistance           = 0.0f;
    lightData.shadowFadeDistance       = 0.0f;
    lightData.shadowBias               = component.shadowBias;
    lightData.castsShadows             = component.castsShadows;
    markLightDirty(lightIndex);
}

void World::setSpotLight(common::EntityId entityId, const SpotLightComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setSpotLight requires valid entity id.");
        return;
    }

    if (!requireAliveEntity(entityId, "setSpotLight"))
    {
        return;
    }

    std::uint32_t lightIndex = 0u;
    if (!tryGetLightIndexForType(entityId, graphics::GpuLightType::Spot, "setSpotLight",
                                 lightIndex))
    {
        return;
    }

    if (lightIndex == kInvalidSlot)
    {
        const std::uint32_t envIndex  = entityEnvironment(entityId);
        const std::uint32_t lightSlot = allocateLightSlot(envIndex, false);
        if (lightSlot == kInvalidSlot)
        {
            return;
        }
        lightIndex                          = envIndex * mSceneLayout.maxLightsPerEnv + lightSlot;
        mRenderLightIndices[entityId]       = lightIndex;
        mRenderLights[lightIndex].entityId  = entityId;
        mRenderLights[lightIndex].envIndex  = envIndex;
        mRenderLights[lightIndex].lightSlot = lightSlot;
    }

    const TransformComponent transform = tryGetTransform(entityId).value_or(TransformComponent{});
    graphics::LightData &lightData     = mRenderLights[lightIndex];
    lightData.type                     = graphics::GpuLightType::Spot;
    lightData.position                 = transform.worldTransform.position;
    lightData.direction                = component.direction;
    lightData.color                    = component.color;
    lightData.intensity                = component.intensity;
    lightData.range                    = component.range;
    lightData.innerConeAngle           = component.innerConeAngle;
    lightData.outerConeAngle           = component.outerConeAngle;
    lightData.shadowDistance           = 0.0f;
    lightData.shadowFadeDistance       = 0.0f;
    lightData.shadowBias               = component.shadowBias;
    lightData.castsShadows             = component.castsShadows;
    markLightDirty(lightIndex);
}

void World::setRigidBody(common::EntityId entityId, const RigidBodyComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setRigidBody requires valid entity id.");
        return;
    }

    if (!requireAliveEntity(entityId, "setRigidBody"))
    {
        return;
    }

    if (!component.simulated)
    {
        if (mPhysicsWorld.removeRigidBody(entityId))
        {
            clearColliderLinks(entityId);
            mPhysicsRenderableMappingsDirty = true;
        }
        mPhysicsLinks[entityId].hasRigidBody = false;
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
    state.environmentIndex        = entityEnvironment(entityId);
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
        clearColliderLinks(entityId);
        mPhysicsRenderableMappingsDirty = true;
    }
    return removed;
}

bool World::setSoftBody(common::EntityId entityId, const SoftBodyComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setSoftBody requires valid entity id.");
        return false;
    }

    if (!requireAliveEntity(entityId, "setSoftBody"))
    {
        return false;
    }

    if (!component.simulated)
    {
        (void)removeSoftBody(entityId);
        return true;
    }

    if (!tryGetMeshRenderer(entityId).has_value())
    {
        CRESSIM_LOG_WARNING("setSoftBody without a mesh renderer on the same entity.");
    }

    TransformComponent transform{};
    if (const std::optional<TransformComponent> t = tryGetTransform(entityId))
    {
        transform = *t;
    }

    physics::SoftBodyState state{};
    state.entityId             = entityId;
    state.environmentIndex     = entityEnvironment(entityId);
    state.source               = component.source;
    state.material             = component.material;
    state.restTransform        = transform.worldTransform;
    state.particleMass         = component.particleMass;
    state.particleRadius       = component.particleRadius;
    state.edgeCompliance       = component.edgeCompliance;
    state.volumeCompliance     = component.volumeCompliance;
    state.simulated            = component.simulated;
    state.selfCollisionEnabled = component.selfCollisionEnabled;
    state.collisionLayer       = component.collisionLayer;
    state.collisionMask        = component.collisionMask;

    if (!mPhysicsWorld.upsertSoftBody(state))
    {
        return false;
    }
    mPhysicsLinks[entityId].hasSoftBody = true;
    mDrawRegistryDirty                  = true;
    mSoftBodyRenderBindingsDirty        = true;
    return true;
}

bool World::removeSoftBody(common::EntityId entityId)
{
    auto it = mPhysicsLinks.find(entityId);
    if (it != mPhysicsLinks.end())
    {
        it->second.hasSoftBody = false;
    }
    if (const auto renderIt = mRenderableIndices.find(entityId);
        renderIt != mRenderableIndices.end())
    {
        markRenderablePoseDirty(static_cast<std::uint32_t>(renderIt->second));
        markRenderableMetadataDirty(static_cast<std::uint32_t>(renderIt->second));
    }
    mDrawRegistryDirty           = true;
    mSoftBodyRenderBindingsDirty = true;
    return mPhysicsWorld.removeSoftBody(entityId);
}

bool World::setFluid(common::EntityId entityId, const FluidComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setFluid requires valid entity id.");
        return false;
    }

    if (!requireAliveEntity(entityId, "setFluid"))
    {
        return false;
    }

    if (!component.simulated)
    {
        (void)removeFluid(entityId);
        return true;
    }

    TransformComponent transform{};
    if (const std::optional<TransformComponent> t = tryGetTransform(entityId))
    {
        transform = *t;
    }

    physics::FluidState state{};
    state.entityId         = entityId;
    state.environmentIndex = entityEnvironment(entityId);
    state.source           = component.source;
    state.material         = component.material;
    state.visualColor      = component.visualColor;
    state.restTransform    = transform.worldTransform;
    state.particleMass     = component.particleMass;
    state.particleRadius   = component.particleRadius;
    state.simulated        = component.simulated;
    state.collisionLayer   = component.collisionLayer;
    state.collisionMask    = component.collisionMask;

    if (!mPhysicsWorld.upsertFluid(state))
    {
        return false;
    }
    mPhysicsLinks[entityId].hasFluid = true;
    mDrawRegistryDirty               = true;
    return true;
}

bool World::removeFluid(common::EntityId entityId)
{
    auto it = mPhysicsLinks.find(entityId);
    if (it != mPhysicsLinks.end())
    {
        it->second.hasFluid = false;
    }
    mDrawRegistryDirty = true;
    return mPhysicsWorld.removeFluid(entityId);
}

World::ColliderHandle World::addCollider(common::EntityId entityId,
                                         const ColliderComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("addCollider requires valid entity id.");
        return {};
    }

    if (!requireAliveEntity(entityId, "addCollider"))
    {
        return {};
    }

    auto body = mPhysicsLinks.find(entityId);
    if (body == mPhysicsLinks.end() || !body->second.hasRigidBody)
    {
        CRESSIM_LOG_ERROR("addCollider requires a rigid body on the entity.");
        return {};
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
    state.staticFriction = component.staticFriction;
    state.restitution    = component.restitution;
    state.collisionLayer = component.collisionLayer;
    state.collisionMask  = component.collisionMask;

    mPhysicsWorld.upsertCollider(state);
    mPhysicsLinks[entityId].colliders.push_back(handle);
    mColliderOwnerEntity[handle.id] = entityId;
    return handle;
}

void World::updateCollider(ColliderHandle handle, const ColliderComponent &component)
{
    if (!handle.isValid())
    {
        CRESSIM_LOG_ERROR("updateCollider requires valid collider handle.");
        return;
    }

    const auto ownerIt = mColliderOwnerEntity.find(handle.id);
    if (ownerIt == mColliderOwnerEntity.end())
    {
        CRESSIM_LOG_ERROR("updateCollider received an unknown collider handle.");
        return;
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
    state.staticFriction = component.staticFriction;
    state.restitution    = component.restitution;
    state.collisionLayer = component.collisionLayer;
    state.collisionMask  = component.collisionMask;

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
        auto &handles = physIt->second.colliders;
        handles.erase(std::remove_if(handles.begin(), handles.end(),
                                     [&](const ColliderHandle h) { return h.id == handle.id; }),
                      handles.end());
    }

    mColliderOwnerEntity.erase(ownerIt);
    return mPhysicsWorld.removeCollider(handle.id);
}

bool World::removeTransform(common::EntityId entityId)
{
    const auto it = mTransformIndex.find(entityId);
    if (it == mTransformIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last  = static_cast<std::uint32_t>(mTransforms.entityIds.size() - 1u);
    const common::EntityId movedEntity = mTransforms.entityIds[last];

    if (index != last)
    {
        mTransforms.entityIds[index]  = mTransforms.entityIds[last];
        mTransforms.components[index] = mTransforms.components[last];
        mTransformIndex[movedEntity]  = index;
    }

    mTransforms.entityIds.pop_back();
    mTransforms.components.pop_back();
    mTransformIndex.erase(it);

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
    const graphics::RenderableInstance &instance = mRenderables[objectIndex];
    reclaimDenseSlot(mFreeRenderableSlotsByEnv, instance.envIndex, instance.objectSlot);
    mRenderables[objectIndex] = {};
    markRenderablePoseDirty(objectIndex);
    markRenderableMetadataDirty(objectIndex);
    mRenderableIndices.erase(it);
    mDrawRegistryDirty              = true;
    mPhysicsRenderableMappingsDirty = true;
    mSoftBodyRenderBindingsDirty    = true;
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
    const graphics::CameraData &camera = mRenderCameras[cameraIndex];
    reclaimDenseSlot(mFreeCameraSlotsByEnv, camera.envIndex, camera.cameraSlot);
    mRenderCameras[cameraIndex] = {};
    mRenderCameraIndices.erase(it);
    markCameraDirty(cameraIndex);
    return true;
}

bool World::removeDirectionalLight(common::EntityId entityId)
{
    const auto component = tryGetDirectionalLight(entityId);
    if (!component.has_value())
    {
        return false;
    }
    return removeLight(entityId);
}

bool World::removePointLight(common::EntityId entityId)
{
    const auto component = tryGetPointLight(entityId);
    if (!component.has_value())
    {
        return false;
    }
    return removeLight(entityId);
}

bool World::removeSpotLight(common::EntityId entityId)
{
    const auto component = tryGetSpotLight(entityId);
    if (!component.has_value())
    {
        return false;
    }
    return removeLight(entityId);
}

std::optional<TransformComponent> World::tryGetTransform(common::EntityId entityId) const
{
    const auto it = mTransformIndex.find(entityId);
    if (it == mTransformIndex.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = it->second;
    return mTransforms.components[index];
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

    const graphics::CameraData &camera = mRenderCameras[static_cast<std::uint32_t>(it->second)];
    CameraComponent component{};
    component.verticalFovDegrees = camera.verticalFovDegrees;
    component.nearClip           = camera.nearClip;
    component.farClip            = camera.farClip;
    component.output             = camera.output;
    component.outputWidth        = camera.outputWidth;
    component.outputHeight       = camera.outputHeight;
    component.viewport           = camera.viewport;
    component.clearColor         = camera.clearColor;
    component.clearDepth         = camera.clearDepth;
    component.clearColorValue    = camera.clearColorValue;
    component.clearDepthValue    = camera.clearDepthValue;
    component.backgroundMode =
        camera.backgroundMode == graphics::CameraBackgroundMode::EnvironmentCubemap
            ? CameraComponent::BackgroundMode::EnvironmentCubemap
            : CameraComponent::BackgroundMode::ClearColor;
    component.renderOrder = camera.renderOrder;
    return component;
}

std::optional<DirectionalLightComponent> World::tryGetDirectionalLight(
    common::EntityId entityId) const
{
    const auto it = mRenderLightIndices.find(entityId);
    if (it == mRenderLightIndices.end())
    {
        return std::nullopt;
    }

    const graphics::LightData &light = mRenderLights[static_cast<std::uint32_t>(it->second)];
    if (light.type != graphics::GpuLightType::Directional)
    {
        return std::nullopt;
    }
    DirectionalLightComponent component{};
    component.direction          = light.direction;
    component.color              = light.color;
    component.intensity          = light.intensity;
    component.shadowDistance     = light.shadowDistance;
    component.shadowFadeDistance = light.shadowFadeDistance;
    component.shadowBias         = light.shadowBias;
    component.castsShadows       = light.castsShadows;
    return component;
}

std::optional<PointLightComponent> World::tryGetPointLight(common::EntityId entityId) const
{
    const auto it = mRenderLightIndices.find(entityId);
    if (it == mRenderLightIndices.end())
    {
        return std::nullopt;
    }

    const graphics::LightData &light = mRenderLights[static_cast<std::uint32_t>(it->second)];
    if (light.type != graphics::GpuLightType::Point)
    {
        return std::nullopt;
    }

    PointLightComponent component{};
    component.color        = light.color;
    component.intensity    = light.intensity;
    component.range        = light.range;
    component.shadowBias   = light.shadowBias;
    component.castsShadows = light.castsShadows;
    return component;
}

std::optional<SpotLightComponent> World::tryGetSpotLight(common::EntityId entityId) const
{
    const auto it = mRenderLightIndices.find(entityId);
    if (it == mRenderLightIndices.end())
    {
        return std::nullopt;
    }

    const graphics::LightData &light = mRenderLights[static_cast<std::uint32_t>(it->second)];
    if (light.type != graphics::GpuLightType::Spot)
    {
        return std::nullopt;
    }

    SpotLightComponent component{};
    component.direction      = light.direction;
    component.color          = light.color;
    component.intensity      = light.intensity;
    component.range          = light.range;
    component.innerConeAngle = light.innerConeAngle;
    component.outerConeAngle = light.outerConeAngle;
    component.shadowBias     = light.shadowBias;
    component.castsShadows   = light.castsShadows;
    return component;
}

std::optional<RigidBodyComponent> World::tryGetRigidBody(common::EntityId entityId) const
{
    const physics::RigidBodyState *rb = mPhysicsWorld.tryGetRigidBody(entityId);
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

std::optional<SoftBodyComponent> World::tryGetSoftBody(common::EntityId entityId) const
{
    const physics::SoftBodyState *softBody = mPhysicsWorld.tryGetSoftBody(entityId);
    if (!softBody)
    {
        return std::nullopt;
    }

    SoftBodyComponent component{};
    component.source               = softBody->source;
    component.material             = softBody->material;
    component.particleMass         = softBody->particleMass;
    component.particleRadius       = softBody->particleRadius;
    component.edgeCompliance       = softBody->edgeCompliance;
    component.volumeCompliance     = softBody->volumeCompliance;
    component.simulated            = softBody->simulated;
    component.selfCollisionEnabled = softBody->selfCollisionEnabled;
    component.collisionLayer       = softBody->collisionLayer;
    component.collisionMask        = softBody->collisionMask;
    return component;
}

std::optional<FluidComponent> World::tryGetFluid(common::EntityId entityId) const
{
    const physics::FluidState *fluid = mPhysicsWorld.tryGetFluid(entityId);
    if (!fluid)
    {
        return std::nullopt;
    }

    FluidComponent component{};
    component.source         = fluid->source;
    component.material       = fluid->material;
    component.visualColor    = fluid->visualColor;
    component.particleMass   = fluid->particleMass;
    component.particleRadius = fluid->particleRadius;
    component.simulated      = fluid->simulated;
    component.collisionLayer = fluid->collisionLayer;
    component.collisionMask  = fluid->collisionMask;
    return component;
}

std::optional<ColliderComponent> World::tryGetCollider(ColliderHandle handle) const
{
    if (!handle.isValid())
    {
        return std::nullopt;
    }

    const physics::ColliderState *c = mPhysicsWorld.tryGetCollider(handle.id);
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
    out.staticFriction = c->staticFriction;
    out.restitution    = c->restitution;
    out.collisionLayer = c->collisionLayer;
    out.collisionMask  = c->collisionMask;
    return out;
}

const std::vector<World::ColliderHandle> &World::colliderHandles(common::EntityId entityId) const
{
    const auto it = mPhysicsLinks.find(entityId);
    return it != mPhysicsLinks.end() ? it->second.colliders : emptyColliderHandleList();
}

physics::PhysicsWorld &World::physicsWorld() noexcept
{
    return mPhysicsWorld;
}

const physics::PhysicsWorld &World::physicsWorld() const noexcept
{
    return mPhysicsWorld;
}

void World::setGpuEntityScene(const graphics::GpuEntitySceneView &sceneView) noexcept
{
    mGpuEntityScene = sceneView;
}

const std::vector<graphics::RenderableInstance> &World::renderables() const noexcept
{
    return mRenderables;
}

const std::vector<graphics::CameraData> &World::cameras() const noexcept
{
    return mRenderCameras;
}

const std::vector<graphics::LightData> &World::lights() const noexcept
{
    return mRenderLights;
}

const std::vector<Diligent::float4> &World::renderObjectPositions() const noexcept
{
    return mRenderObjectPositions;
}

const std::vector<Diligent::float4> &World::renderObjectOrientations() const noexcept
{
    return mRenderObjectOrientations;
}

const std::vector<Diligent::float4> &World::renderObjectScales() const noexcept
{
    return mRenderObjectScales;
}

const std::vector<graphics::GpuRenderableMetadata> &World::renderableMetadata() const noexcept
{
    return mRenderableMetadataHost;
}

const std::vector<graphics::GpuRenderableQueueInfo> &World::renderableQueueInfo() const noexcept
{
    return mRenderableQueueInfoHost;
}

const std::vector<graphics::GpuCameraInput> &World::cameraInputs() const noexcept
{
    return mCameraInputsHost;
}

const std::vector<graphics::GpuLightInput> &World::lightInputs() const noexcept
{
    return mLightInputsHost;
}

const std::vector<graphics::GpuLocalLightSelection> &World::localLightSelections() const noexcept
{
    return mLocalLightSelectionsHost;
}

const std::vector<graphics::GpuSoftBodyVertexBinding> &World::softBodyVertexBindings()
    const noexcept
{
    return mSoftBodyVertexBindingsHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry> &World::opaqueDrawRegistry()
    const noexcept
{
    return mOpaqueDrawRegistryHost;
}

const std::vector<graphics::TransparentDrawEntry> &World::transparentDrawRegistry() const noexcept
{
    return mTransparentDrawRegistryHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry> &World::shadowDrawRegistry()
    const noexcept
{
    return mShadowDrawRegistryHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry> &World::localShadowDrawRegistry()
    const noexcept
{
    return mLocalShadowDrawRegistryHost;
}

const std::vector<EntityPoseMappingEntry> &World::physicsRenderableMappings()
{
    const std::uint64_t rigidBodyTopologyRevision = mPhysicsWorld.rigidBodyTopologyRevision();
    if (!mPhysicsRenderableMappingsDirty &&
        mCachedPhysicsRenderableMappingsBodyTopologyRevision == rigidBodyTopologyRevision)
    {
        return mPhysicsRenderableMappingsCache;
    }

    mPhysicsRenderableMappingsCache.clear();

    const auto &rigidBodies = mPhysicsWorld.rigidBodySoA();
    if (!rigidBodies.entityIds.empty() && !mRenderables.empty())
    {
        std::unordered_map<common::EntityId, std::uint32_t> rigidBodyIndexByEntity;
        rigidBodyIndexByEntity.reserve(rigidBodies.entityIds.size());
        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(rigidBodies.entityIds.size()); ++i)
        {
            rigidBodyIndexByEntity.emplace(rigidBodies.entityIds[i], i);
        }

        mPhysicsRenderableMappingsCache.reserve(mRenderables.size());
        for (const graphics::RenderableInstance &renderable : mRenderables)
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

            EntityPoseMappingEntry entry{};
            entry.sourcePoseIndex = rigidBodyIt->second;
            entry.objectIndex     = renderable.envIndex * mSceneLayout.maxRenderableObjectsPerEnv +
                                    renderable.objectSlot;
            mPhysicsRenderableMappingsCache.push_back(entry);
        }
    }

    mPhysicsRenderableMappingsDirty                      = false;
    mCachedPhysicsRenderableMappingsBodyTopologyRevision = rigidBodyTopologyRevision;
    return mPhysicsRenderableMappingsCache;
}

const graphics::GpuEntitySceneView &World::gpuEntityScene() const noexcept
{
    return mGpuEntityScene;
}

graphics::HostSceneView World::hostSceneView() const noexcept
{
    return graphics::HostSceneView{
        &mRenderables,
        &mRenderableMetadataHost,
        &mRenderCameras,
        &mRenderLights,
        &mEnvironmentIbls,
        &mEnvironmentFluids,
        &mOpaqueDrawRegistryHost,
        &mTransparentDrawRegistryHost,
        &mShadowDrawRegistryHost,
        &mLocalShadowDrawRegistryHost,
        &mGpuEntityScene,
    };
}

void World::ensureRenderStateUpToDate(const graphics::RenderResourceManager &resources)
{
    const std::uint64_t softBodyTopologyRevision = mPhysicsWorld.softBodyTopologyRevision();
    if (mCachedSoftBodyRenderTopologyRevision != softBodyTopologyRevision)
    {
        mSoftBodyRenderBindingsDirty          = true;
        mCachedSoftBodyRenderTopologyRevision = softBodyTopologyRevision;
    }

    const std::uint64_t physicsRevision = mPhysicsWorld.authoredRevision();
    if (mCachedSoftBodyPhysicsRevision != physicsRevision)
    {
        for (std::uint32_t objectIndex = 0u;
             objectIndex < static_cast<std::uint32_t>(mRenderables.size()); ++objectIndex)
        {
            const graphics::RenderableInstance &renderable = mRenderables[objectIndex];
            if (renderable.entityId == common::kInvalidEntityId ||
                renderable.objectSlot == kInvalidSlot)
            {
                continue;
            }
            const auto physIt = mPhysicsLinks.find(renderable.entityId);
            if (physIt != mPhysicsLinks.end() && physIt->second.hasSoftBody)
            {
                markRenderablePoseDirty(objectIndex);
                markRenderableMetadataDirty(objectIndex);
            }
        }
        mCachedSoftBodyPhysicsRevision = physicsRevision;
    }

    if (mSoftBodyRenderBindingsDirty)
    {
        mPhysicsWorld.ensureSoftBodyDerivedStateUpToDate();
        rebuildSoftBodyRenderBindings(resources);
    }

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
        refreshLightEntry(lightIndex);
    }

    if (!mDirtyLightIndices.empty())
    {
        rebuildLocalLightSelections();
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

void World::rebuildSoftBodyRenderBindings(const graphics::RenderResourceManager &resources)
{
    std::fill(mSoftBodyVertexBindingBaseByObject.begin(), mSoftBodyVertexBindingBaseByObject.end(),
              kInvalidSlot);
    std::fill(mSoftBodyVertexNormalBaseByObject.begin(), mSoftBodyVertexNormalBaseByObject.end(),
              kInvalidSlot);
    std::fill(mSoftBodyVertexCountByObject.begin(), mSoftBodyVertexCountByObject.end(), 0u);
    mSoftBodyVertexBindingsHost.clear();
    physics::SoftRenderDataHost softRenderData;
    const std::vector<physics::SoftBodyState> &softBodies = mPhysicsWorld.softBodySnapshot();
    softRenderData.softBodyParticleRanges.resize(softBodies.size(), Diligent::uint2{0u, 0u});
    for (std::uint32_t softBodyIndex = 0u;
         softBodyIndex < static_cast<std::uint32_t>(softBodies.size()); ++softBodyIndex)
    {
        const physics::SoftBodyState &softBody = softBodies[softBodyIndex];
        softRenderData.softBodyParticleRanges[softBodyIndex] =
            Diligent::uint2{softBody.particleOffset, softBody.particleCount};
    }

    for (std::uint32_t objectIndex = 0u;
         objectIndex < static_cast<std::uint32_t>(mRenderables.size()); ++objectIndex)
    {
        const graphics::RenderableInstance &renderable = mRenderables[objectIndex];
        if (renderable.entityId == common::kInvalidEntityId ||
            renderable.objectSlot == kInvalidSlot)
        {
            continue;
        }

        const auto physIt = mPhysicsLinks.find(renderable.entityId);
        if (physIt == mPhysicsLinks.end() || !physIt->second.hasSoftBody)
        {
            continue;
        }

        const physics::SoftBodyState *softBody = mPhysicsWorld.tryGetSoftBody(renderable.entityId);
        const graphics::MeshResourceDesc *mesh = resources.tryGetMesh(renderable.mesh);
        if (softBody == nullptr || mesh == nullptr || mesh->vertices.empty() ||
            softBody->restPositions.empty())
        {
            CRESSIM_LOG_ERROR("Soft-body render binding build failed for entity ",
                              renderable.entityId,
                              ": missing visual mesh or soft-body rest-position data.");
            continue;
        }

        std::unordered_map<QuantizedPointKey, std::uint32_t, QuantizedPointKeyHash>
            restIndexByQuantizedPosition;
        restIndexByQuantizedPosition.reserve(softBody->restPositions.size());
        std::vector<Diligent::float3> restPositionsLocal;
        restPositionsLocal.reserve(softBody->restPositions.size());
        for (std::uint32_t localParticleIndex = 0u;
             localParticleIndex < static_cast<std::uint32_t>(softBody->restPositions.size());
             ++localParticleIndex)
        {
            const Diligent::float3 localRestPosition = inverseTransformPoint(
                softBody->restTransform, softBody->restPositions[localParticleIndex]);
            restPositionsLocal.push_back(localRestPosition);
            restIndexByQuantizedPosition.try_emplace(
                quantizePoint(localRestPosition, kSoftBodyVertexMatchEpsilon), localParticleIndex);
        }

        const std::uint32_t bindingBase =
            static_cast<std::uint32_t>(mSoftBodyVertexBindingsHost.size());
        const std::uint32_t normalBase = bindingBase;
        const std::uint32_t rangeBase =
            static_cast<std::uint32_t>(softRenderData.vertexTriangleRanges.size());
        bool valid = true;
        std::vector<std::vector<std::uint32_t>> incidentTriangles(mesh->vertices.size());
        for (const graphics::MeshResourceDesc::Vertex &vertex : mesh->vertices)
        {
            const std::optional<std::uint32_t> matchedRestIndex = findMatchingRestVertexLocal(
                vertex.position, restPositionsLocal, restIndexByQuantizedPosition,
                kSoftBodyVertexMatchEpsilon);
            if (!matchedRestIndex.has_value())
            {
                valid = false;
                break;
            }

            graphics::GpuSoftBodyVertexBinding binding{};
            binding.particleIndex = softBody->particleOffset + matchedRestIndex.value();
            mSoftBodyVertexBindingsHost.push_back(binding);
            const Diligent::float3 fallbackNormal =
                transformNormal(softBody->restTransform, vertex.normal);
            softRenderData.fallbackNormals.emplace_back(fallbackNormal.x, fallbackNormal.y,
                                                        fallbackNormal.z, 0.0f);
            softRenderData.vertexTriangleRanges.push_back({});
        }

        if (!valid)
        {
            CRESSIM_LOG_ERROR("Soft-body render binding build failed for entity ",
                              renderable.entityId,
                              ": visual mesh vertices must match tet rest vertices exactly.");
            mSoftBodyVertexBindingsHost.resize(bindingBase);
            softRenderData.fallbackNormals.resize(normalBase);
            softRenderData.vertexTriangleRanges.resize(rangeBase);
            continue;
        }

        const std::uint32_t triangleBase =
            static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size());
        for (std::size_t triangleIndex = 0u; triangleIndex + 2u < mesh->indices.size();
             triangleIndex += 3u)
        {
            const std::uint32_t i0 = mesh->indices[triangleIndex + 0u];
            const std::uint32_t i1 = mesh->indices[triangleIndex + 1u];
            const std::uint32_t i2 = mesh->indices[triangleIndex + 2u];
            if (i0 >= mesh->vertices.size() || i1 >= mesh->vertices.size() ||
                i2 >= mesh->vertices.size())
            {
                continue;
            }

            const std::uint32_t particleIndex0 =
                mSoftBodyVertexBindingsHost[bindingBase + i0].particleIndex;
            const std::uint32_t particleIndex1 =
                mSoftBodyVertexBindingsHost[bindingBase + i1].particleIndex;
            const std::uint32_t particleIndex2 =
                mSoftBodyVertexBindingsHost[bindingBase + i2].particleIndex;
            const std::uint32_t renderTriangleIndex =
                static_cast<std::uint32_t>(softRenderData.triangleParticleIndices.size());
            softRenderData.triangleParticleIndices.emplace_back(particleIndex0, particleIndex1,
                                                                particleIndex2, 0u);
            incidentTriangles[i0].push_back(renderTriangleIndex);
            incidentTriangles[i1].push_back(renderTriangleIndex);
            incidentTriangles[i2].push_back(renderTriangleIndex);
        }

        (void)triangleBase;
        for (std::uint32_t localVertexIndex = 0u;
             localVertexIndex < static_cast<std::uint32_t>(incidentTriangles.size());
             ++localVertexIndex)
        {
            physics::SoftRenderVertexTriangleRange &range =
                softRenderData.vertexTriangleRanges[rangeBase + localVertexIndex];
            range.start = static_cast<std::uint32_t>(softRenderData.vertexTriangleIndices.size());
            range.count = static_cast<std::uint32_t>(incidentTriangles[localVertexIndex].size());
            softRenderData.vertexTriangleIndices.insert(softRenderData.vertexTriangleIndices.end(),
                                                        incidentTriangles[localVertexIndex].begin(),
                                                        incidentTriangles[localVertexIndex].end());
        }

        mSoftBodyVertexBindingBaseByObject[objectIndex] = bindingBase;
        mSoftBodyVertexNormalBaseByObject[objectIndex]  = normalBase;
        mSoftBodyVertexCountByObject[objectIndex] =
            static_cast<std::uint32_t>(mesh->vertices.size());
        markRenderableMetadataDirty(objectIndex);
    }

    mSoftBodyRenderBindingsDirty = false;
    mPhysicsWorld.setSoftRenderData(softRenderData);
}

void World::refreshDirtyRenderableMetadata(const graphics::RenderResourceManager &resources)
{
    // TODO: some metadata depends on resources too, not just world state:
    // if a mesh or material changes later inside RenderResourceManager, World does
    // not currently know which object slots should become dirty.
    // Currently no supported mutation path exists, but if resources become mutable, World has no
    // dependency tracking to invalidate affected slots.

    for (const std::uint32_t objectIndex : mDirtyRenderableMetadataIndices)
    {
        if (objectIndex >= mRenderableMetadataHost.size() || objectIndex >= mRenderables.size())
        {
            continue;
        }

        const graphics::RenderableInstance &renderable = mRenderables[objectIndex];
        graphics::GpuRenderableMetadata entry{};
        graphics::GpuRenderableFlags renderableFlags = graphics::GpuRenderableFlags::None;
        entry.softBodyVertexBindingBase              = kInvalidSlot;
        entry.softBodyVertexNormalBase               = kInvalidSlot;
        entry.softBodyIndex                          = kInvalidSlot;
        entry.softBodyVertexCount                    = 0u;
        if (renderable.entityId != common::kInvalidEntityId &&
            renderable.objectSlot != kInvalidSlot)
        {
            if (renderable.visible)
            {
                renderableFlags |= graphics::GpuRenderableFlags::Active;
            }

            const graphics::MaterialResourceDesc *material =
                resources.tryGetMaterial(renderable.material);
            if (material != nullptr && renderable.visible && !rendersInTransparentPass(*material))
            {
                renderableFlags |= graphics::GpuRenderableFlags::Opaque;
                if (material->castsShadows)
                {
                    renderableFlags |= graphics::GpuRenderableFlags::ShadowCaster;
                }
            }

            Diligent::float3 localBoundsMin{};
            Diligent::float3 localBoundsMax{};
            bool hasBounds    = false;
            const auto physIt = mPhysicsLinks.find(renderable.entityId);
            if (physIt != mPhysicsLinks.end() && physIt->second.hasSoftBody)
            {
                const physics::SoftBodyState *softBody =
                    mPhysicsWorld.tryGetSoftBody(renderable.entityId);
                if (softBody != nullptr)
                {
                    const std::vector<physics::SoftBodyState> &softBodies =
                        mPhysicsWorld.softBodySnapshot();
                    for (std::uint32_t softBodyIndex = 0u;
                         softBodyIndex < static_cast<std::uint32_t>(softBodies.size());
                         ++softBodyIndex)
                    {
                        if (softBodies[softBodyIndex].entityId == renderable.entityId)
                        {
                            entry.softBodyIndex = softBodyIndex;
                            break;
                        }
                    }
                    // If render binding generation failed, metadata can still carry a soft-body
                    // index while the vertex binding bases remain invalid. That currently makes
                    // culling use soft-body bounds while vertex shaders fall back to undeformed
                    // data.
                    localBoundsMin = Diligent::float3{0.0f, 0.0f, 0.0f};
                    localBoundsMax = Diligent::float3{0.0f, 0.0f, 0.0f};
                    hasBounds      = true;
                }
                entry.softBodyVertexBindingBase = mSoftBodyVertexBindingBaseByObject[objectIndex];
                entry.softBodyVertexNormalBase  = mSoftBodyVertexNormalBaseByObject[objectIndex];
                entry.softBodyVertexCount       = mSoftBodyVertexCountByObject[objectIndex];
            }

            if (!hasBounds &&
                resources.tryGetMeshLocalBounds(renderable.mesh, localBoundsMin, localBoundsMax))
            {
                entry.localBoundsMin =
                    Diligent::float4{localBoundsMin.x, localBoundsMin.y, localBoundsMin.z, 1.0f};
                entry.localBoundsMax =
                    Diligent::float4{localBoundsMax.x, localBoundsMax.y, localBoundsMax.z, 1.0f};
                hasBounds = true;
            }
            if (hasBounds)
            {
                entry.localBoundsMin =
                    Diligent::float4{localBoundsMin.x, localBoundsMin.y, localBoundsMin.z, 1.0f};
                entry.localBoundsMax =
                    Diligent::float4{localBoundsMax.x, localBoundsMax.y, localBoundsMax.z, 1.0f};
            }
        }
        entry.flags = static_cast<std::uint32_t>(renderableFlags);

        mRenderableMetadataHost[objectIndex] = entry;
    }
}

void World::rebuildDrawRegistries(const graphics::RenderResourceManager &resources)
{
    // TODO: mesh/material resources still authored on CPU.
    // I don't know if we can let GPU do this part completely.

    std::map<DrawBucketKey, std::vector<std::uint32_t>> opaqueObjectsByKey;
    std::map<DrawBucketKey, std::vector<std::uint32_t>> shadowObjectsByKey;
    mRenderableQueueInfoHost.assign(mRenderables.size(), graphics::GpuRenderableQueueInfo{});
    mOpaqueDrawRegistryHost.clear();
    mTransparentDrawRegistryHost.clear();
    mShadowDrawRegistryHost.clear();
    mLocalShadowDrawRegistryHost.clear();

    for (std::uint32_t objectIndex = 0u;
         objectIndex < static_cast<std::uint32_t>(mRenderables.size()); ++objectIndex)
    {
        const graphics::RenderableInstance &renderable = mRenderables[objectIndex];
        if (renderable.entityId == common::kInvalidEntityId ||
            renderable.objectSlot == kInvalidSlot || !renderable.visible)
        {
            continue;
        }

        const graphics::MeshResourceDesc *mesh = resources.tryGetMesh(renderable.mesh);
        const graphics::MaterialResourceDesc *material =
            resources.tryGetMaterial(renderable.material);
        if (mesh == nullptr || material == nullptr || mesh->vertices.empty() ||
            mesh->indices.size() < 3)
        {
            continue;
        }
        const std::int32_t renderOrder = material->renderOrder;
        const graphics::MaterialFeatureFlags materialFeatureFlags =
            graphics::effectiveMaterialFeatureFlags(*material);

        const auto physIt = mPhysicsLinks.find(renderable.entityId);
        const graphics::MaterialProgramFamily programFamily =
            (physIt != mPhysicsLinks.end() && physIt->second.hasSoftBody)
                ? graphics::MaterialProgramFamily::SoftBodyLit
                : material->pipeline.programFamily;

        const DrawBucketKey key{
            renderOrder,
            programFamily,
            static_cast<std::uint32_t>(materialFeatureFlags),
            renderable.material.id,
            renderable.mesh.id,
        };
        if (rendersInTransparentPass(*material))
        {
            graphics::TransparentDrawEntry transparentEntry{};
            transparentEntry.renderOrder                      = renderOrder;
            transparentEntry.drawCommand.programFamily        = key.programFamily;
            transparentEntry.drawCommand.materialFeatureFlags = materialFeatureFlags;
            transparentEntry.drawCommand.meshId               = key.meshId;
            transparentEntry.drawCommand.materialId           = key.materialId;
            transparentEntry.drawCommand.meshVersion =
                resources.meshVersion(graphics::MeshHandle{key.meshId});
            transparentEntry.drawCommand.indexCount =
                static_cast<std::uint32_t>(mesh->indices.size());
            transparentEntry.objectIndex = objectIndex;
            mTransparentDrawRegistryHost.push_back(transparentEntry);
            continue;
        }
        opaqueObjectsByKey[key].push_back(objectIndex);
        if (material->castsShadows)
        {
            shadowObjectsByKey[key].push_back(objectIndex);
        }
    }

    std::uint32_t opaqueCommandIndex = 0u;
    for (const auto &[key, objectIndices] : opaqueObjectsByKey)
    {
        if (objectIndices.empty())
        {
            continue;
        }

        const graphics::MeshResourceDesc *mesh =
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

    std::uint32_t shadowCommandBaseIndex  = 0u;
    std::uint32_t localShadowCommandIndex = 0u;
    for (const auto &[key, objectIndices] : shadowObjectsByKey)
    {
        if (objectIndices.empty())
        {
            continue;
        }

        const graphics::MeshResourceDesc *mesh =
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
            mLocalShadowDrawRegistryHost.push_back(entry);
        }

        for (const std::uint32_t objectIndex : objectIndices)
        {
            mRenderableQueueInfoHost[objectIndex].shadowCommandBaseIndex  = shadowCommandBaseIndex;
            mRenderableQueueInfoHost[objectIndex].localShadowCommandIndex = localShadowCommandIndex;
        }
        shadowCommandBaseIndex += graphics::kShadowCascadeCount;
        ++localShadowCommandIndex;
    }
}

void World::refreshRenderablePose(std::uint32_t objectIndex)
{
    if (objectIndex >= mRenderables.size())
    {
        return;
    }

    graphics::RenderableInstance &renderable = mRenderables[objectIndex];
    if (renderable.entityId == common::kInvalidEntityId || renderable.objectSlot == kInvalidSlot)
    {
        mRenderObjectPositions[objectIndex]    = Diligent::float4{0.0f, 0.0f, 0.0f, 0.0f};
        mRenderObjectOrientations[objectIndex] = Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f};
        mRenderObjectScales[objectIndex]       = Diligent::float4{1.0f, 1.0f, 1.0f, 0.0f};
        return;
    }

    const auto physIt = mPhysicsLinks.find(renderable.entityId);
    if (physIt != mPhysicsLinks.end() && physIt->second.hasSoftBody)
    {
        renderable.worldTransform =
            tryGetTransform(renderable.entityId).value_or(TransformComponent{}).worldTransform;
        mRenderObjectPositions[objectIndex] = Diligent::float4{
            renderable.worldTransform.position.x, renderable.worldTransform.position.y,
            renderable.worldTransform.position.z, 1.0f};
        mRenderObjectOrientations[objectIndex] = Diligent::float4{
            renderable.worldTransform.rotation.q.x, renderable.worldTransform.rotation.q.y,
            renderable.worldTransform.rotation.q.z, renderable.worldTransform.rotation.q.w};
        mRenderObjectScales[objectIndex] =
            Diligent::float4{renderable.worldTransform.scale.x, renderable.worldTransform.scale.y,
                             renderable.worldTransform.scale.z, 0.0f};
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

    graphics::CameraData &cameraData = mRenderCameras[cameraIndex];
    if (cameraData.entityId == common::kInvalidEntityId || cameraData.cameraSlot == kInvalidSlot)
    {
        mCameraInputsHost[cameraIndex] = {};
        return;
    }

    cameraData.worldTransform =
        tryGetTransform(cameraData.entityId).value_or(TransformComponent{}).worldTransform;

    graphics::GpuCameraInput input{};
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

void World::refreshLightEntry(std::uint32_t lightIndex)
{
    if (lightIndex >= mRenderLights.size())
    {
        return;
    }

    graphics::LightData &lightData = mRenderLights[lightIndex];
    if (lightData.entityId == common::kInvalidEntityId || lightData.lightSlot == kInvalidSlot)
    {
        mLightInputsHost[lightIndex] = {};
        return;
    }

    const TransformComponent transform =
        tryGetTransform(lightData.entityId).value_or(TransformComponent{});
    refreshTransformDerivedLightState(lightData, transform);

    graphics::GpuLightInput input{};
    input.positionRange      = Diligent::float4{lightData.position.x, lightData.position.y,
                                                lightData.position.z, lightData.range};
    input.directionIntensity = Diligent::float4{lightData.direction.x, lightData.direction.y,
                                                lightData.direction.z, lightData.intensity};
    input.color = Diligent::float4{lightData.color.x, lightData.color.y, lightData.color.z, 0.0f};
    const float innerConeRadians = lightData.innerConeAngle * 0.01745329251994329577f;
    const float outerConeRadians = lightData.outerConeAngle * 0.01745329251994329577f;
    input.spotAngles     = Diligent::float4{std::cos(innerConeRadians), std::cos(outerConeRadians),
                                            lightData.innerConeAngle, lightData.outerConeAngle};
    input.shadowDistance = lightData.shadowDistance;
    input.shadowFadeDistance     = lightData.shadowFadeDistance;
    input.shadowBias             = lightData.shadowBias;
    input.envIndex               = lightData.envIndex;
    input.lightSlot              = lightData.lightSlot;
    input.type                   = static_cast<std::uint32_t>(lightData.type);
    input.active                 = 1u;
    input.castsShadows           = lightData.castsShadows ? 1u : 0u;
    mLightInputsHost[lightIndex] = input;
}

void World::rebuildLocalLightSelections()
{
    mLocalLightSelectionsHost.assign(mSceneLayout.envCount, graphics::GpuLocalLightSelection{});
    for (std::uint32_t lightIndex = 0u;
         lightIndex < static_cast<std::uint32_t>(mRenderLights.size()); ++lightIndex)
    {
        const graphics::LightData &light = mRenderLights[lightIndex];
        if (!isValidLight(light) || light.envIndex >= mLocalLightSelectionsHost.size() ||
            light.lightSlot == graphics::kMainDirectionalLightSlot)
        {
            continue;
        }

        const bool active =
            light.intensity > 0.0f &&
            (light.type != graphics::GpuLightType::Directional ||
             (light.direction.x * light.direction.x + light.direction.y * light.direction.y +
              light.direction.z * light.direction.z) > 1.0e-6f);
        if (!active)
        {
            continue;
        }

        graphics::GpuLocalLightSelection &selection = mLocalLightSelectionsHost[light.envIndex];
        const std::uint32_t nextCount               = selection.localLightCount;
        if (nextCount < graphics::kForwardLocalLightCap)
        {
            selection.lightIndices[nextCount] = lightIndex;
            selection.localLightCount         = nextCount + 1u;
        }

        if (!light.castsShadows)
        {
            continue;
        }

        if (light.type == graphics::GpuLightType::Point)
        {
            if (selection.shadowedPointLightCount < graphics::kShadowedPointLightCap)
            {
                ++selection.shadowedPointLightCount;
            }
        }
        else if (selection.shadowedLocalLightCount < graphics::kShadowedLocalLightCap)
        {
            ++selection.shadowedLocalLightCount;
        }
    }
}

bool World::moveRenderableToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    const auto indexIt = mRenderableIndices.find(entityId);
    if (indexIt == mRenderableIndices.end())
    {
        return true;
    }

    const std::uint32_t oldObjectIndex      = static_cast<std::uint32_t>(indexIt->second);
    graphics::RenderableInstance renderable = mRenderables[oldObjectIndex];
    const std::uint32_t newObjectSlot =
        allocateDenseSlot(mFreeRenderableSlotsByEnv, mNextRenderableSlotByEnv, envIndex,
                          mSceneLayout.maxRenderableObjectsPerEnv, "renderable");
    if (newObjectSlot == kInvalidSlot)
    {
        CRESSIM_LOG_ERROR("setEntityEnvironment failed to move renderable for entity ", entityId,
                          " into environment ", envIndex, ".");
        return false;
    }

    reclaimDenseSlot(mFreeRenderableSlotsByEnv, renderable.envIndex, renderable.objectSlot);
    mRenderables[oldObjectIndex] = {};
    renderable.envIndex          = envIndex;
    renderable.objectSlot        = newObjectSlot;
    const std::uint32_t newObjectIndex =
        envIndex * mSceneLayout.maxRenderableObjectsPerEnv + newObjectSlot;
    mRenderables[newObjectIndex] = renderable;
    markRenderableMetadataDirty(oldObjectIndex);
    markRenderableMetadataDirty(newObjectIndex);
    markRenderablePoseDirty(oldObjectIndex);
    markRenderablePoseDirty(newObjectIndex);
    indexIt->second              = newObjectIndex;
    mSoftBodyRenderBindingsDirty = true;
    return true;
}

bool World::moveCameraToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    const auto indexIt = mRenderCameraIndices.find(entityId);
    if (indexIt == mRenderCameraIndices.end())
    {
        return true;
    }

    const std::uint32_t oldCameraIndex = static_cast<std::uint32_t>(indexIt->second);
    graphics::CameraData camera        = mRenderCameras[oldCameraIndex];
    const std::uint32_t newCameraSlot =
        allocateDenseSlot(mFreeCameraSlotsByEnv, mNextCameraSlotByEnv, envIndex,
                          mSceneLayout.maxCamerasPerEnv, "camera");
    if (newCameraSlot == kInvalidSlot)
    {
        CRESSIM_LOG_ERROR("setEntityEnvironment failed to move camera for entity ", entityId,
                          " into environment ", envIndex, ".");
        return false;
    }

    reclaimDenseSlot(mFreeCameraSlotsByEnv, camera.envIndex, camera.cameraSlot);
    mRenderCameras[oldCameraIndex]     = {};
    camera.envIndex                    = envIndex;
    camera.cameraSlot                  = newCameraSlot;
    const std::uint32_t newCameraIndex = envIndex * mSceneLayout.maxCamerasPerEnv + newCameraSlot;
    mRenderCameras[newCameraIndex]     = camera;
    indexIt->second                    = newCameraIndex;
    markCameraDirty(oldCameraIndex);
    markCameraDirty(newCameraIndex);
    return true;
}

bool World::moveLightToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    const auto indexIt = mRenderLightIndices.find(entityId);
    if (indexIt == mRenderLightIndices.end())
    {
        return true;
    }

    const std::uint32_t oldLightIndex = static_cast<std::uint32_t>(indexIt->second);
    graphics::LightData light         = mRenderLights[oldLightIndex];
    const std::uint32_t newLightSlot =
        allocateLightSlot(envIndex, light.type == graphics::GpuLightType::Directional);
    if (newLightSlot == kInvalidSlot)
    {
        CRESSIM_LOG_ERROR("setEntityEnvironment failed to move light for entity ", entityId,
                          " into environment ", envIndex, ".");
        return false;
    }

    reclaimDenseSlot(mFreeLightSlotsByEnv, light.envIndex, light.lightSlot);
    mRenderLights[oldLightIndex]      = {};
    light.envIndex                    = envIndex;
    light.lightSlot                   = newLightSlot;
    const std::uint32_t newLightIndex = envIndex * mSceneLayout.maxLightsPerEnv + newLightSlot;
    mRenderLights[newLightIndex]      = light;
    indexIt->second                   = newLightIndex;
    markLightDirty(oldLightIndex);
    markLightDirty(newLightIndex);
    return true;
}

bool World::isLightSlotOccupied(std::uint32_t envIndex, std::uint32_t slot) const noexcept
{
    if (slot >= mSceneLayout.maxLightsPerEnv)
    {
        return false;
    }

    const std::uint32_t lightIndex = envIndex * mSceneLayout.maxLightsPerEnv + slot;
    return lightIndex < mRenderLights.size() && isValidLight(mRenderLights[lightIndex]);
}

std::uint32_t World::allocateLightSlot(std::uint32_t envIndex, bool reserveMainDirectionalSlot)
{
    if (mSceneLayout.maxLightsPerEnv == 0u)
    {
        CRESSIM_LOG_ERROR("light capacity exceeded for environment ", envIndex, ".");
        return kInvalidSlot;
    }

    if (reserveMainDirectionalSlot &&
        !isLightSlotOccupied(envIndex, graphics::kMainDirectionalLightSlot))
    {
        auto &freeSlots = mFreeLightSlotsByEnv[envIndex];
        const auto it =
            std::find(freeSlots.begin(), freeSlots.end(), graphics::kMainDirectionalLightSlot);
        if (it != freeSlots.end())
        {
            freeSlots.erase(it);
        }
        auto &nextSlot = mNextLightSlotByEnv[envIndex];
        nextSlot       = std::max(nextSlot, graphics::kMainDirectionalLightSlot + 1u);
        return graphics::kMainDirectionalLightSlot;
    }

    auto &freeSlots = mFreeLightSlotsByEnv[envIndex];
    for (auto it = freeSlots.begin(); it != freeSlots.end(); ++it)
    {
        if (*it == graphics::kMainDirectionalLightSlot)
        {
            continue;
        }
        const std::uint32_t slot = *it;
        freeSlots.erase(it);
        return slot;
    }

    auto &nextSlot = mNextLightSlotByEnv[envIndex];
    if (nextSlot <= graphics::kMainDirectionalLightSlot)
    {
        nextSlot = graphics::kMainDirectionalLightSlot + 1u;
    }
    if (nextSlot >= mSceneLayout.maxLightsPerEnv)
    {
        CRESSIM_LOG_ERROR("light capacity exceeded for environment ", envIndex, ".");
        return kInvalidSlot;
    }
    return nextSlot++;
}

bool World::tryGetLightIndexForType(common::EntityId entityId, graphics::GpuLightType type,
                                    const char *operation, std::uint32_t &lightIndex) const noexcept
{
    const auto indexIt = mRenderLightIndices.find(entityId);
    if (indexIt == mRenderLightIndices.end())
    {
        lightIndex = kInvalidSlot;
        return true;
    }

    const std::uint32_t existingLightIndex = static_cast<std::uint32_t>(indexIt->second);
    if (existingLightIndex >= mRenderLights.size())
    {
        lightIndex = kInvalidSlot;
        return true;
    }

    const graphics::LightData &existingLight = mRenderLights[existingLightIndex];
    if (!isValidLight(existingLight))
    {
        lightIndex = kInvalidSlot;
        return true;
    }

    if (existingLight.type != type)
    {
        CRESSIM_LOG_ERROR(operation, " rejected for entity ", entityId,
                          ": each entity may own at most one light type.");
        return false;
    }

    lightIndex = existingLightIndex;
    return true;
}

bool World::removeLight(common::EntityId entityId)
{
    const auto it = mRenderLightIndices.find(entityId);
    if (it == mRenderLightIndices.end())
    {
        return false;
    }

    const std::uint32_t lightIndex   = static_cast<std::uint32_t>(it->second);
    const graphics::LightData &light = mRenderLights[lightIndex];
    reclaimDenseSlot(mFreeLightSlotsByEnv, light.envIndex, light.lightSlot);
    mRenderLights[lightIndex] = {};
    mRenderLightIndices.erase(it);
    markLightDirty(lightIndex);
    return true;
}

void World::clearColliderLinks(common::EntityId entityId) noexcept
{
    const auto physIt = mPhysicsLinks.find(entityId);
    if (physIt == mPhysicsLinks.end())
    {
        return;
    }

    for (const ColliderHandle handle : physIt->second.colliders)
    {
        mColliderOwnerEntity.erase(handle.id);
    }
    physIt->second.colliders.clear();
}

} // namespace cressim::neo::engine
