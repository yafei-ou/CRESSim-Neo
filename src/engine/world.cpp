#include "engine/world.h"
#include "common/logger.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_set>

namespace cressim::neo::engine
{

namespace
{
constexpr std::uint32_t kInvalidSlot        = 0xffffffffu;
constexpr float kSoftBodyVertexMatchEpsilon = 1.0e-3f;

void bumpGeneration(std::uint64_t &generation) noexcept
{
    ++generation;
    if (generation == 0u)
    {
        generation = 1u;
    }
}

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

physics::ColliderState makeColliderState(const common::EntityId entityId,
                                         const physics::ColliderId colliderId,
                                         const ColliderComponent &component)
{
    physics::ColliderState state{};
    state.colliderId     = colliderId;
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
    return state;
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

graphics::GpuSoftBodyVertexBinding makeExactSoftBodyVertexBinding(
    std::uint32_t particleIndex) noexcept
{
    graphics::GpuSoftBodyVertexBinding binding{};
    binding.particleIndices =
        Diligent::uint4{particleIndex, particleIndex, particleIndex, particleIndex};
    binding.weights = Diligent::float4{1.0f, 0.0f, 0.0f, 0.0f};
    return binding;
}

graphics::GpuSoftBodyVertexBinding makeNearestParticleSkinBinding(
    const Diligent::float3 &visualVertexLocal,
    const std::vector<Diligent::float3> &restPositionsLocal, std::uint32_t particleOffset) noexcept
{
    struct Candidate
    {
        float distanceSq         = std::numeric_limits<float>::max();
        std::uint32_t localIndex = 0u;
        bool valid               = false;
    };

    std::array<Candidate, 4u> nearest{};
    for (std::uint32_t localParticleIndex = 0u;
         localParticleIndex < static_cast<std::uint32_t>(restPositionsLocal.size());
         ++localParticleIndex)
    {
        const Diligent::float3 delta = restPositionsLocal[localParticleIndex] - visualVertexLocal;
        const float distanceSq       = Diligent::dot(delta, delta);
        for (std::size_t slot = 0u; slot < nearest.size(); ++slot)
        {
            if (nearest[slot].valid && distanceSq >= nearest[slot].distanceSq)
            {
                continue;
            }

            for (std::size_t moveSlot = nearest.size() - 1u; moveSlot > slot; --moveSlot)
            {
                nearest[moveSlot] = nearest[moveSlot - 1u];
            }
            nearest[slot] = Candidate{distanceSq, localParticleIndex, true};
            break;
        }
    }

    graphics::GpuSoftBodyVertexBinding binding{};
    const std::uint32_t fallbackParticleIndex =
        particleOffset + (nearest[0].valid ? nearest[0].localIndex : 0u);
    binding.particleIndices = Diligent::uint4{fallbackParticleIndex, fallbackParticleIndex,
                                              fallbackParticleIndex, fallbackParticleIndex};
    binding.weights         = Diligent::float4{1.0f, 0.0f, 0.0f, 0.0f};
    if (!nearest[0].valid)
    {
        return binding;
    }

    std::array<float, 4u> rawWeights{};
    float weightSum                  = 0.0f;
    constexpr float kDistanceEpsilon = 1.0e-6f;
    for (std::size_t slot = 0u; slot < nearest.size(); ++slot)
    {
        if (!nearest[slot].valid)
        {
            continue;
        }

        const std::uint32_t particleIndex = particleOffset + nearest[slot].localIndex;
        const float distance              = std::sqrt(std::max(nearest[slot].distanceSq, 0.0f));
        const float rawWeight             = 1.0f / (distance + kDistanceEpsilon);
        rawWeights[slot]                  = rawWeight;
        weightSum += rawWeight;

        if (slot == 0u)
        {
            binding.particleIndices.x = particleIndex;
        }
        else if (slot == 1u)
        {
            binding.particleIndices.y = particleIndex;
        }
        else if (slot == 2u)
        {
            binding.particleIndices.z = particleIndex;
        }
        else
        {
            binding.particleIndices.w = particleIndex;
        }
    }

    if (weightSum <= 0.0f)
    {
        return binding;
    }

    binding.weights = Diligent::float4{rawWeights[0] / weightSum, rawWeights[1] / weightSum,
                                       rawWeights[2] / weightSum, rawWeights[3] / weightSum};
    return binding;
}

std::uint32_t dominantParticleIndex(const graphics::GpuSoftBodyVertexBinding &binding) noexcept
{
    std::uint32_t particleIndex = binding.particleIndices.x;
    float weight                = binding.weights.x;
    if (binding.weights.y > weight)
    {
        particleIndex = binding.particleIndices.y;
        weight        = binding.weights.y;
    }
    if (binding.weights.z > weight)
    {
        particleIndex = binding.particleIndices.z;
        weight        = binding.weights.z;
    }
    if (binding.weights.w > weight)
    {
        particleIndex = binding.particleIndices.w;
    }
    return particleIndex;
}

} // namespace

struct World::Impl
{
    static constexpr std::uint32_t kInvalidIndex = 0xffffffffu;

    explicit Impl(World &owner) : mOwner(&owner) {}

    // Some implementation helpers reuse the facade's public query semantics.
    World *mOwner = nullptr;

    [[nodiscard]] bool requireAliveEntity(common::EntityId entityId,
                                          const char *operation) const noexcept;
    void ensureHostSceneStorage();
    std::uint32_t ensureEntityPoseSlot(common::EntityId entityId);
    void releaseEntityPoseSlot(common::EntityId entityId) noexcept;
    void markEntityPoseDirty(common::EntityId entityId);
    void refreshEntityPoseSlot(std::uint32_t entityPoseSlot);
    void refreshRenderablePose(std::uint32_t objectIndex);
    void refreshCameraEntry(std::uint32_t cameraIndex);
    void refreshLightEntry(std::uint32_t lightIndex);
    void rebuildLocalLightSelections();
    void rebuildSoftBodyRenderBindings(const graphics::RenderResourceManager &resources);
    void rebuildCurveRenderBindings(const graphics::RenderResourceManager &resources);
    void refreshDirtyRenderableMetadata(const graphics::RenderResourceManager &resources);
    void rebuildDrawRegistries(const graphics::RenderResourceManager &resources);
    void clearDirtyIndexSet(std::vector<std::uint32_t> &dirtyIndices,
                            std::vector<std::uint8_t> &dirtyBits);
    void markRenderablePoseDirty(std::uint32_t objectIndex);
    void markCameraDirty(std::uint32_t cameraIndex);
    void markLightDirty(std::uint32_t lightIndex);
    void markRenderableMetadataDirty(std::uint32_t objectIndex);
    [[nodiscard]] bool moveRenderableToEnvironment(common::EntityId entityId,
                                                   std::uint32_t envIndex);
    [[nodiscard]] bool moveCameraToEnvironment(common::EntityId entityId, std::uint32_t envIndex);
    [[nodiscard]] bool moveLightToEnvironment(common::EntityId entityId, std::uint32_t envIndex);
    void clearColliderLinks(common::EntityId entityId) noexcept;
    [[nodiscard]] bool tryGetLightIndexForType(common::EntityId entityId,
                                               graphics::GpuLightType type, const char *operation,
                                               std::uint32_t &lightIndex) const noexcept;
    [[nodiscard]] bool removeLight(common::EntityId entityId);
    [[nodiscard]] bool isLightSlotOccupied(std::uint32_t envIndex,
                                           std::uint32_t slot) const noexcept;
    [[nodiscard]] std::uint32_t allocateLightSlot(std::uint32_t envIndex,
                                                  bool reserveMainDirectionalSlot);

    struct PhysicsLink
    {
        bool hasRigidBody = false;
        bool hasSoftBody  = false;
        bool hasStrand    = false;
        bool hasFluid     = false;
        std::vector<ColliderHandle> colliders;
    };

    struct TransformStorage
    {
        std::vector<common::EntityId> entityIds;
        std::vector<TransformComponent> components;
    };

    common::EntityId mNextEntityId = 1;
    std::uint32_t mNextColliderId  = 1;
    std::vector<common::EntityId> mEntities;
    std::unordered_set<common::EntityId> mAlive;
    TransformStorage mTransforms{};
    std::unordered_map<common::EntityId, std::uint32_t> mTransformIndex{};
    std::vector<common::EntityId> mEntityPoseEntities{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityPoseSlotByEntity{};
    std::vector<std::uint32_t> mFreeEntityPoseSlots{};
    std::unordered_map<common::EntityId, PhysicsLink> mPhysicsLinks{};
    std::unordered_map<std::uint32_t, common::EntityId> mColliderOwnerEntity{};
    std::unordered_map<common::EntityId, UltrasoundProbeComponent> mUltrasoundProbes{};
    std::unordered_map<common::EntityId, UltrasoundRendererComponent> mUltrasoundRenderers{};
    std::unordered_map<common::EntityId, ProceduralDeformableCurveRenderComponent>
        mProceduralDeformableCurveRenders{};
    std::unordered_map<common::EntityId, UltrasoundScattererSourceComponent>
        mUltrasoundScattererSources{};
    std::unordered_map<common::EntityId, std::vector<UltrasoundAmplitudeRange>>
        mUltrasoundScattererAmplitudeRanges{};
    std::unordered_map<common::EntityId, UltrasoundProbeResult> mUltrasoundProbeResults{};
    std::uint64_t mUltrasoundScattererAmplitudeRevision = 0u;
    physics::PhysicsWorld mPhysicsWorld{};
    common::SceneLayoutDesc mSceneLayout{};
    bool mWorldSceneAuthored = false;
    std::unordered_map<common::EntityId, std::uint32_t> mEntityEnvironments{};
    std::vector<graphics::RenderableInstance> mRenderables{};
    std::vector<graphics::CameraData> mRenderCameras{};
    std::vector<graphics::LightData> mRenderLights{};
    std::vector<Diligent::float4> mEntityPosePositionsHost{};
    std::vector<Diligent::float4> mEntityPoseOrientationsHost{};
    std::vector<Diligent::float4> mEntityPoseScalesHost{};
    std::vector<Diligent::float4> mRenderObjectPositions{};
    std::vector<Diligent::float4> mRenderObjectOrientations{};
    std::vector<Diligent::float4> mRenderObjectScales{};
    std::vector<graphics::GpuRenderableMetadata> mRenderableMetadataHost{};
    std::vector<graphics::GpuRenderableQueueInfo> mRenderableQueueInfoHost{};
    std::vector<graphics::GpuCameraInput> mCameraInputsHost{};
    std::vector<graphics::GpuLightInput> mLightInputsHost{};
    std::vector<graphics::GpuLocalLightSelection> mLocalLightSelectionsHost{};
    std::vector<graphics::GpuSoftBodyVertexBinding> mSoftBodyVertexBindingsHost{};
    std::vector<graphics::EnvironmentIblDesc> mEnvironmentIbls{};
    std::vector<graphics::EnvironmentFluidDesc> mEnvironmentFluids{};
    std::vector<graphics::IndirectCommandRegistryEntry> mOpaqueDrawRegistryHost{};
    std::vector<graphics::TransparentDrawEntry> mTransparentDrawRegistryHost{};
    std::vector<graphics::IndirectCommandRegistryEntry> mShadowDrawRegistryHost{};
    std::vector<graphics::IndirectCommandRegistryEntry> mLocalShadowDrawRegistryHost{};
    graphics::GpuEntitySceneView mGpuEntityScene{};
    std::vector<EntityPoseMappingEntry> mPhysicsRenderableMappingsCache{};
    std::unordered_map<common::EntityId, std::size_t> mRenderableIndices{};
    std::unordered_map<common::EntityId, std::size_t> mRenderCameraIndices{};
    std::unordered_map<common::EntityId, std::size_t> mRenderLightIndices{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextRenderableSlotByEnv{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextCameraSlotByEnv{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextLightSlotByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeRenderableSlotsByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeCameraSlotsByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeLightSlotsByEnv{};
    std::vector<std::uint32_t> mDirtyEntityPoseSlots{};
    std::vector<std::uint8_t> mDirtyEntityPoseBits{};
    std::vector<std::uint32_t> mDirtyRenderablePoseIndices{};
    std::vector<std::uint8_t> mDirtyRenderablePoseBits{};
    std::vector<std::uint32_t> mDirtyRenderableMetadataIndices{};
    std::vector<std::uint8_t> mDirtyRenderableMetadataBits{};
    std::vector<std::uint32_t> mDirtyCameraIndices{};
    std::vector<std::uint8_t> mDirtyCameraBits{};
    std::vector<std::uint32_t> mDirtyLightIndices{};
    std::vector<std::uint8_t> mDirtyLightBits{};
    bool mDrawRegistryDirty              = true;
    bool mPhysicsRenderableMappingsDirty = true;
    bool mSoftBodyRenderBindingsDirty    = true;
    bool mCurveRenderBindingsDirty       = true;
    std::vector<std::uint32_t> mSoftBodyVertexBindingBaseByObject{};
    std::vector<std::uint32_t> mSoftBodyVertexNormalBaseByObject{};
    std::vector<std::uint32_t> mSoftBodyVertexCountByObject{};
    std::vector<std::uint32_t> mCurveRenderVertexBaseByObject{};
    std::vector<std::uint32_t> mCurveRenderVertexNormalBaseByObject{};
    std::vector<std::uint32_t> mCurveRenderIndexByObject{};
    std::vector<std::uint32_t> mCurveRenderVertexCountByObject{};
    std::uint64_t mCachedPhysicsRenderableMappingsBodyTopologyRevision = ~0ull;
    std::uint64_t mCachedSoftBodyRenderTopologyRevision                = ~0ull;
    std::uint64_t mCachedSoftBodyPhysicsRevision                       = ~0ull;
    std::uint64_t mCachedCurveRenderPhysicsRevision                    = ~0ull;
    std::uint64_t mEntityPoseRevision                                  = 1u;
    std::uint64_t mRenderableMetadataRevision                          = 1u;
    std::uint64_t mRenderableQueueInfoRevision                         = 1u;
    std::uint64_t mSoftBodyVertexBindingRevision                       = 1u;
    std::uint64_t mCameraInputRevision                                 = 1u;
    std::uint64_t mLightInputRevision                                  = 1u;
    std::uint64_t mLocalLightSelectionRevision                         = 1u;
};

World::World() : mImpl(std::make_unique<Impl>(*this)) {}

World::~World() = default;

World::World(const World &other) : mImpl(std::make_unique<Impl>(*other.mImpl))
{
    mImpl->mOwner = this;
}

World &World::operator=(const World &other)
{
    if (this != &other)
    {
        mImpl         = std::make_unique<Impl>(*other.mImpl);
        mImpl->mOwner = this;
    }
    return *this;
}

World::World(World &&other) noexcept : mImpl(std::move(other.mImpl))
{
    if (mImpl)
    {
        mImpl->mOwner = this;
    }
}

World &World::operator=(World &&other) noexcept
{
    if (this != &other)
    {
        mImpl = std::move(other.mImpl);
        if (mImpl)
        {
            mImpl->mOwner = this;
        }
    }
    return *this;
}

common::EntityId World::createEntity(std::uint32_t envIndex)
{
    // A globally unique EntityId within this World.

    if (envIndex >= mImpl->mSceneLayout.envCount)
    {
        CRESSIM_LOG_ERROR("createEntity envIndex exceeds configured environment count.");
        return common::kInvalidEntityId;
    }
    mImpl->ensureHostSceneStorage();
    mImpl->mWorldSceneAuthored = true;

    const common::EntityId entityId = mImpl->mNextEntityId++;
    mImpl->mAlive.insert(entityId);
    mImpl->mEntities.push_back(entityId);
    mImpl->mEntityEnvironments[entityId] = envIndex;
    return entityId;
}

bool World::destroyEntity(common::EntityId entityId)
{
    if (mImpl->mAlive.erase(entityId) == 0)
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
    removeStrand(entityId);
    removeProceduralDeformableCurveRender(entityId);
    removeFluid(entityId);
    removeUltrasoundProbe(entityId);
    removeUltrasoundRenderer(entityId);
    removeUltrasoundScattererSource(entityId);

    auto physIt = mImpl->mPhysicsLinks.find(entityId);
    if (physIt != mImpl->mPhysicsLinks.end())
    {
        const auto colliders = physIt->second.colliders;
        for (const ColliderHandle h : colliders)
        {
            removeCollider(h);
        }
        mImpl->mPhysicsLinks.erase(physIt);
    }

    mImpl->mEntities.erase(std::remove(mImpl->mEntities.begin(), mImpl->mEntities.end(), entityId),
                           mImpl->mEntities.end());
    mImpl->releaseEntityPoseSlot(entityId);
    mImpl->mEntityEnvironments.erase(entityId);
    return true;
}

void World::setSceneLayout(const common::SceneLayoutDesc &layout)
{
    if (mImpl->mWorldSceneAuthored)
    {
        CRESSIM_LOG_ERROR("cannot re-configure scene layout after authoring.");
        return;
    }

    mImpl->mSceneLayout = layout;
    mImpl->ensureHostSceneStorage();
}

const common::SceneLayoutDesc &World::sceneLayout() const noexcept
{
    return mImpl->mSceneLayout;
}

bool World::setEntityEnvironment(common::EntityId entityId, std::uint32_t envIndex)
{
    if (!isAlive(entityId))
    {
        return false;
    }
    if (envIndex >= mImpl->mSceneLayout.envCount)
    {
        CRESSIM_LOG_ERROR("setEntityEnvironment envIndex exceeds configured environment count.");
        return false;
    }

    const std::uint32_t previousEnv = entityEnvironment(entityId);
    if (previousEnv == envIndex)
    {
        return true;
    }

    const auto physIt = mImpl->mPhysicsLinks.find(entityId);
    if (physIt != mImpl->mPhysicsLinks.end())
    {
        if (physIt->second.hasRigidBody)
        {
            if (physics::RigidBodyState *rigidBody = mImpl->mPhysicsWorld.tryGetRigidBody(entityId))
            {
                physics::RigidBodyState updated = *rigidBody;
                updated.environmentIndex        = envIndex;
                mImpl->mPhysicsWorld.upsertRigidBody(updated);
            }
        }
        if (physIt->second.hasSoftBody)
        {
            if (physics::SoftBodyState *softBody = mImpl->mPhysicsWorld.tryGetSoftBody(entityId))
            {
                physics::SoftBodyState updated = *softBody;
                updated.environmentIndex       = envIndex;
                if (!mImpl->mPhysicsWorld.upsertSoftBody(updated))
                {
                    return false;
                }
            }
        }
        if (physIt->second.hasStrand)
        {
            if (physics::StrandState *strand = mImpl->mPhysicsWorld.tryGetStrand(entityId))
            {
                physics::StrandState updated = *strand;
                updated.environmentIndex     = envIndex;
                if (!mImpl->mPhysicsWorld.upsertStrand(updated))
                {
                    return false;
                }
            }
        }
        if (physIt->second.hasFluid)
        {
            if (physics::FluidState *fluid = mImpl->mPhysicsWorld.tryGetFluid(entityId))
            {
                physics::FluidState updated = *fluid;
                updated.environmentIndex    = envIndex;
                if (!mImpl->mPhysicsWorld.upsertFluid(updated))
                {
                    return false;
                }
            }
        }
    }
    mImpl->mEntityEnvironments[entityId] = envIndex;
    if (!mImpl->moveRenderableToEnvironment(entityId, envIndex) ||
        !mImpl->moveCameraToEnvironment(entityId, envIndex) ||
        !mImpl->moveLightToEnvironment(entityId, envIndex))
    {
        mImpl->mEntityEnvironments[entityId] = previousEnv;
        if (physIt != mImpl->mPhysicsLinks.end())
        {
            if (physIt->second.hasRigidBody)
            {
                if (physics::RigidBodyState *rigidBody =
                        mImpl->mPhysicsWorld.tryGetRigidBody(entityId))
                {
                    physics::RigidBodyState reverted = *rigidBody;
                    reverted.environmentIndex        = previousEnv;
                    mImpl->mPhysicsWorld.upsertRigidBody(reverted);
                }
            }
            if (physIt->second.hasSoftBody)
            {
                if (physics::SoftBodyState *softBody =
                        mImpl->mPhysicsWorld.tryGetSoftBody(entityId))
                {
                    physics::SoftBodyState reverted = *softBody;
                    reverted.environmentIndex       = previousEnv;
                    if (!mImpl->mPhysicsWorld.upsertSoftBody(reverted))
                    {
                        CRESSIM_LOG_ERROR("setEntityEnvironment failed to restore previous "
                                          "soft-body environment for entity ",
                                          entityId, ".");
                    }
                }
            }
            if (physIt->second.hasStrand)
            {
                if (physics::StrandState *strand = mImpl->mPhysicsWorld.tryGetStrand(entityId))
                {
                    physics::StrandState reverted = *strand;
                    reverted.environmentIndex     = previousEnv;
                    if (!mImpl->mPhysicsWorld.upsertStrand(reverted))
                    {
                        CRESSIM_LOG_ERROR("setEntityEnvironment failed to restore previous "
                                          "strand environment for entity ",
                                          entityId, ".");
                    }
                }
            }
            if (physIt->second.hasFluid)
            {
                if (physics::FluidState *fluid = mImpl->mPhysicsWorld.tryGetFluid(entityId))
                {
                    physics::FluidState reverted = *fluid;
                    reverted.environmentIndex    = previousEnv;
                    if (!mImpl->mPhysicsWorld.upsertFluid(reverted))
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
    mImpl->mDrawRegistryDirty              = true;
    mImpl->mPhysicsRenderableMappingsDirty = true;
    return true;
}

std::uint32_t World::entityEnvironment(common::EntityId entityId) const noexcept
{
    const auto it = mImpl->mEntityEnvironments.find(entityId);
    return it != mImpl->mEntityEnvironments.end() ? it->second : 0u;
}

bool World::setEnvironmentIbl(std::uint32_t envIndex, const graphics::EnvironmentIblDesc &desc)
{
    if (envIndex >= mImpl->mSceneLayout.envCount)
    {
        return false;
    }

    // In case user did not explicitly set the layout.
    mImpl->ensureHostSceneStorage();

    // Setting authoring status to true prevents scene layout change after this.
    mImpl->mWorldSceneAuthored = true;

    mImpl->mEnvironmentIbls[envIndex] = desc;
    return true;
}

const graphics::EnvironmentIblDesc *World::tryGetEnvironmentIbl(
    std::uint32_t envIndex) const noexcept
{
    if (envIndex >= mImpl->mEnvironmentIbls.size())
    {
        return nullptr;
    }
    return &mImpl->mEnvironmentIbls[envIndex];
}

bool World::setEnvironmentFluid(std::uint32_t envIndex, const graphics::EnvironmentFluidDesc &desc)
{
    if (envIndex >= mImpl->mSceneLayout.envCount)
    {
        return false;
    }

    mImpl->ensureHostSceneStorage();
    mImpl->mWorldSceneAuthored          = true;
    mImpl->mEnvironmentFluids[envIndex] = desc;
    return true;
}

const graphics::EnvironmentFluidDesc *World::tryGetEnvironmentFluid(
    std::uint32_t envIndex) const noexcept
{
    if (envIndex >= mImpl->mEnvironmentFluids.size())
    {
        return nullptr;
    }
    return &mImpl->mEnvironmentFluids[envIndex];
}

bool World::isAlive(common::EntityId entityId) const
{
    return mImpl->mAlive.find(entityId) != mImpl->mAlive.end();
}

const std::vector<common::EntityId> &World::entities() const noexcept
{
    return mImpl->mEntities;
}

bool World::Impl::requireAliveEntity(common::EntityId entityId,
                                     const char *operation) const noexcept
{
    if (mAlive.find(entityId) != mAlive.end())
    {
        return true;
    }

    CRESSIM_LOG_ERROR(operation, " requires an existing entity id (entityId=", entityId, ").");
    return false;
}

void World::Impl::ensureHostSceneStorage()
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
        mCurveRenderVertexBaseByObject.assign(objectCapacity, kInvalidSlot);
        mCurveRenderVertexNormalBaseByObject.assign(objectCapacity, kInvalidSlot);
        mCurveRenderIndexByObject.assign(objectCapacity, kInvalidSlot);
        mCurveRenderVertexCountByObject.assign(objectCapacity, 0u);
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
        mCurveRenderBindingsDirty       = true;
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
        mEnvironmentFluids.resize(mSceneLayout.envCount, graphics::defaultEnvironmentFluidDesc());
    }
    if (mLocalLightSelectionsHost.size() != mSceneLayout.envCount)
    {
        mLocalLightSelectionsHost.assign(mSceneLayout.envCount, graphics::GpuLocalLightSelection{});
    }

    if (mDirtyEntityPoseBits.size() != mEntityPoseEntities.size())
    {
        mDirtyEntityPoseBits.resize(mEntityPoseEntities.size(), 0u);
    }
}

std::uint32_t World::Impl::ensureEntityPoseSlot(common::EntityId entityId)
{
    const auto existing = mEntityPoseSlotByEntity.find(entityId);
    if (existing != mEntityPoseSlotByEntity.end())
    {
        return existing->second;
    }

    std::uint32_t slot = kInvalidSlot;
    if (!mFreeEntityPoseSlots.empty())
    {
        slot = mFreeEntityPoseSlots.back();
        mFreeEntityPoseSlots.pop_back();
        mEntityPoseEntities[slot] = entityId;
    }
    else
    {
        slot = static_cast<std::uint32_t>(mEntityPoseEntities.size());
        mEntityPoseEntities.push_back(entityId);
        mEntityPosePositionsHost.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f});
        mEntityPoseOrientationsHost.push_back(Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f});
        mEntityPoseScalesHost.push_back(Diligent::float4{1.0f, 1.0f, 1.0f, 0.0f});
        mDirtyEntityPoseBits.push_back(0u);
    }

    mEntityPoseSlotByEntity.emplace(entityId, slot);
    markEntityPoseDirty(entityId);
    bumpGeneration(mEntityPoseRevision);
    return slot;
}

void World::Impl::releaseEntityPoseSlot(common::EntityId entityId) noexcept
{
    const auto it = mEntityPoseSlotByEntity.find(entityId);
    if (it == mEntityPoseSlotByEntity.end())
    {
        return;
    }

    const std::uint32_t slot = it->second;
    if (slot < mEntityPoseEntities.size())
    {
        mEntityPoseEntities[slot] = common::kInvalidEntityId;
    }
    if (slot < mEntityPosePositionsHost.size())
    {
        mEntityPosePositionsHost[slot] = Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f};
    }
    if (slot < mEntityPoseOrientationsHost.size())
    {
        mEntityPoseOrientationsHost[slot] = Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f};
    }
    if (slot < mEntityPoseScalesHost.size())
    {
        mEntityPoseScalesHost[slot] = Diligent::float4{1.0f, 1.0f, 1.0f, 0.0f};
    }
    if (slot < mDirtyEntityPoseBits.size())
    {
        enqueueDenseDirtyIndex(slot, mDirtyEntityPoseSlots, mDirtyEntityPoseBits);
    }
    mFreeEntityPoseSlots.push_back(slot);
    mEntityPoseSlotByEntity.erase(it);
    mPhysicsRenderableMappingsDirty = true;
    bumpGeneration(mEntityPoseRevision);
}

void World::Impl::markEntityPoseDirty(common::EntityId entityId)
{
    const auto it = mEntityPoseSlotByEntity.find(entityId);
    if (it == mEntityPoseSlotByEntity.end())
    {
        return;
    }

    const std::uint32_t slot = it->second;
    if (slot >= mDirtyEntityPoseBits.size())
    {
        mDirtyEntityPoseBits.resize(slot + 1u, 0u);
    }
    enqueueDenseDirtyIndex(slot, mDirtyEntityPoseSlots, mDirtyEntityPoseBits);
}

void World::Impl::refreshEntityPoseSlot(std::uint32_t entityPoseSlot)
{
    if (entityPoseSlot >= mEntityPoseEntities.size() ||
        entityPoseSlot >= mEntityPosePositionsHost.size() ||
        entityPoseSlot >= mEntityPoseOrientationsHost.size() ||
        entityPoseSlot >= mEntityPoseScalesHost.size())
    {
        return;
    }

    const common::EntityId entityId = mEntityPoseEntities[entityPoseSlot];
    common::Transform transform{};
    if (entityId != common::kInvalidEntityId)
    {
        transform = mOwner->tryGetTransform(entityId).value_or(TransformComponent{}).worldTransform;
    }

    const Diligent::float4 position{transform.position.x, transform.position.y,
                                    transform.position.z, 1.0f};
    const Diligent::float4 orientation{transform.rotation.q.x, transform.rotation.q.y,
                                       transform.rotation.q.z, transform.rotation.q.w};
    const Diligent::float4 scale{transform.scale.x, transform.scale.y, transform.scale.z, 0.0f};

    if (mEntityPosePositionsHost[entityPoseSlot].x != position.x ||
        mEntityPosePositionsHost[entityPoseSlot].y != position.y ||
        mEntityPosePositionsHost[entityPoseSlot].z != position.z ||
        mEntityPoseOrientationsHost[entityPoseSlot].x != orientation.x ||
        mEntityPoseOrientationsHost[entityPoseSlot].y != orientation.y ||
        mEntityPoseOrientationsHost[entityPoseSlot].z != orientation.z ||
        mEntityPoseOrientationsHost[entityPoseSlot].w != orientation.w ||
        mEntityPoseScalesHost[entityPoseSlot].x != scale.x ||
        mEntityPoseScalesHost[entityPoseSlot].y != scale.y ||
        mEntityPoseScalesHost[entityPoseSlot].z != scale.z)
    {
        mEntityPosePositionsHost[entityPoseSlot]    = position;
        mEntityPoseOrientationsHost[entityPoseSlot] = orientation;
        mEntityPoseScalesHost[entityPoseSlot]       = scale;
        bumpGeneration(mEntityPoseRevision);
        return;
    }

    mEntityPosePositionsHost[entityPoseSlot]    = position;
    mEntityPoseOrientationsHost[entityPoseSlot] = orientation;
    mEntityPoseScalesHost[entityPoseSlot]       = scale;
}

void World::Impl::markRenderableMetadataDirty(std::uint32_t objectIndex)
{
    enqueueDenseDirtyIndex(objectIndex, mDirtyRenderableMetadataIndices,
                           mDirtyRenderableMetadataBits);
}

void World::Impl::markRenderablePoseDirty(std::uint32_t objectIndex)
{
    enqueueDenseDirtyIndex(objectIndex, mDirtyRenderablePoseIndices, mDirtyRenderablePoseBits);
}

void World::Impl::markCameraDirty(std::uint32_t cameraIndex)
{
    enqueueDenseDirtyIndex(cameraIndex, mDirtyCameraIndices, mDirtyCameraBits);
}

void World::Impl::markLightDirty(std::uint32_t lightIndex)
{
    enqueueDenseDirtyIndex(lightIndex, mDirtyLightIndices, mDirtyLightBits);
}

void World::Impl::clearDirtyIndexSet(std::vector<std::uint32_t> &dirtyIndices,
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

    if (!mImpl->requireAliveEntity(entityId, "setTransform"))
    {
        return;
    }

    const auto it = mImpl->mTransformIndex.find(entityId);
    if (it == mImpl->mTransformIndex.end())
    {
        const std::uint32_t newIndex =
            static_cast<std::uint32_t>(mImpl->mTransforms.entityIds.size());
        mImpl->mTransforms.entityIds.push_back(entityId);
        mImpl->mTransforms.components.push_back(component);
        mImpl->mTransformIndex.emplace(entityId, newIndex);
    }
    else
    {
        mImpl->mTransforms.components[it->second] = component;
    }

    mImpl->ensureEntityPoseSlot(entityId);
    mImpl->markEntityPoseDirty(entityId);

    // Forcefully set rigid body transform from world transform. This teleports the body (no physics
    // integration). Users should not set the transform direction on an entity with a kinematic
    // rigid body.
    if (auto *rb = mImpl->mPhysicsWorld.tryGetRigidBody(entityId))
    {
        rb->position = component.worldTransform.position;
        rb->rotation = component.worldTransform.rotation;
        rb->scale    = component.worldTransform.scale;
        mImpl->mPhysicsWorld.upsertRigidBody(*rb);
    }
    if (auto *softBody = mImpl->mPhysicsWorld.tryGetSoftBody(entityId))
    {
        physics::SoftBodyState updated = *softBody;
        updated.restTransform          = component.worldTransform;
        if (!mImpl->mPhysicsWorld.upsertSoftBody(updated))
        {
            CRESSIM_LOG_ERROR("setTransform failed to rebuild soft body for entity ", entityId,
                              ".");
        }
    }
    if (auto *fluid = mImpl->mPhysicsWorld.tryGetFluid(entityId))
    {
        physics::FluidState updated = *fluid;
        updated.restTransform       = component.worldTransform;
        if (!mImpl->mPhysicsWorld.upsertFluid(updated))
        {
            CRESSIM_LOG_ERROR("setTransform failed to rebuild fluid for entity ", entityId, ".");
        }
    }
    if (const auto it = mImpl->mRenderableIndices.find(entityId);
        it != mImpl->mRenderableIndices.end())
    {
        mImpl->markRenderablePoseDirty(static_cast<std::uint32_t>(it->second));
    }
    if (const auto it = mImpl->mRenderCameraIndices.find(entityId);
        it != mImpl->mRenderCameraIndices.end())
    {
        mImpl->markCameraDirty(static_cast<std::uint32_t>(it->second));
    }
    if (const auto it = mImpl->mRenderLightIndices.find(entityId);
        it != mImpl->mRenderLightIndices.end())
    {
        mImpl->markLightDirty(static_cast<std::uint32_t>(it->second));
    }
}

void World::setMeshRenderer(common::EntityId entityId, const MeshRendererComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setMeshRenderer requires valid entity id.");
        return;
    }

    if (!mImpl->requireAliveEntity(entityId, "setMeshRenderer"))
    {
        return;
    }

    mImpl->ensureEntityPoseSlot(entityId);

    const auto indexIt        = mImpl->mRenderableIndices.find(entityId);
    std::uint32_t objectIndex = 0u;
    if (indexIt == mImpl->mRenderableIndices.end())
    {
        const std::uint32_t envIndex   = entityEnvironment(entityId);
        const std::uint32_t objectSlot = allocateDenseSlot(
            mImpl->mFreeRenderableSlotsByEnv, mImpl->mNextRenderableSlotByEnv, envIndex,
            mImpl->mSceneLayout.maxRenderableObjectsPerEnv, "renderable");
        if (objectSlot == kInvalidSlot)
        {
            return;
        }
        objectIndex = envIndex * mImpl->mSceneLayout.maxRenderableObjectsPerEnv + objectSlot;
        mImpl->mRenderableIndices[entityId]         = objectIndex;
        mImpl->mRenderables[objectIndex].entityId   = entityId;
        mImpl->mRenderables[objectIndex].envIndex   = envIndex;
        mImpl->mRenderables[objectIndex].objectSlot = objectSlot;
    }
    else
    {
        objectIndex = static_cast<std::uint32_t>(indexIt->second);
    }

    graphics::RenderableInstance &renderable = mImpl->mRenderables[objectIndex];
    renderable.mesh                          = component.mesh;
    renderable.material                      = component.material;
    renderable.segmentationId                = component.segmentationId;
    renderable.visible                       = component.visible;
    mImpl->markRenderableMetadataDirty(objectIndex);
    mImpl->markRenderablePoseDirty(objectIndex);
    mImpl->mDrawRegistryDirty              = true;
    mImpl->mPhysicsRenderableMappingsDirty = true;
    mImpl->mSoftBodyRenderBindingsDirty    = true;
    mImpl->mCurveRenderBindingsDirty       = true;
}

void World::setCamera(common::EntityId entityId, const CameraComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setCamera requires valid entity id.");
        return;
    }

    if (!mImpl->requireAliveEntity(entityId, "setCamera"))
    {
        return;
    }

    mImpl->ensureEntityPoseSlot(entityId);

    const auto indexIt        = mImpl->mRenderCameraIndices.find(entityId);
    std::uint32_t cameraIndex = 0u;
    if (indexIt == mImpl->mRenderCameraIndices.end())
    {
        const std::uint32_t envIndex = entityEnvironment(entityId);
        const std::uint32_t cameraSlot =
            allocateDenseSlot(mImpl->mFreeCameraSlotsByEnv, mImpl->mNextCameraSlotByEnv, envIndex,
                              mImpl->mSceneLayout.maxCamerasPerEnv, "camera");
        if (cameraSlot == kInvalidSlot)
        {
            return;
        }
        cameraIndex = envIndex * mImpl->mSceneLayout.maxCamerasPerEnv + cameraSlot;
        mImpl->mRenderCameraIndices[entityId]         = cameraIndex;
        mImpl->mRenderCameras[cameraIndex].entityId   = entityId;
        mImpl->mRenderCameras[cameraIndex].envIndex   = envIndex;
        mImpl->mRenderCameras[cameraIndex].cameraSlot = cameraSlot;
    }
    else
    {
        cameraIndex = static_cast<std::uint32_t>(indexIt->second);
    }

    graphics::CameraData &cameraData = mImpl->mRenderCameras[cameraIndex];
    cameraData.worldTransform =
        tryGetTransform(entityId).value_or(TransformComponent{}).worldTransform;
    cameraData.verticalFovDegrees = component.verticalFovDegrees;
    cameraData.nearClip           = component.nearClip;
    cameraData.farClip            = component.farClip;
    switch (component.product)
    {
    case CameraComponent::Product::Depth:
        cameraData.product = graphics::CameraData::Product::Depth;
        break;
    case CameraComponent::Product::SegmentationDepth:
        cameraData.product = graphics::CameraData::Product::SegmentationDepth;
        break;
    case CameraComponent::Product::ColorDepth:
    default:
        cameraData.product = graphics::CameraData::Product::ColorDepth;
        break;
    }
    cameraData.output          = component.output;
    cameraData.outputWidth     = component.outputWidth;
    cameraData.outputHeight    = component.outputHeight;
    cameraData.viewport        = component.viewport;
    cameraData.clearColor      = component.clearColor;
    cameraData.clearDepth      = component.clearDepth;
    cameraData.clearColorValue = component.clearColorValue;
    cameraData.clearDepthValue = component.clearDepthValue;
    cameraData.backgroundMode =
        component.backgroundMode == CameraComponent::BackgroundMode::EnvironmentCubemap
            ? graphics::CameraBackgroundMode::EnvironmentCubemap
            : graphics::CameraBackgroundMode::ClearColor;
    cameraData.renderOrder = component.renderOrder;
    mImpl->markCameraDirty(cameraIndex);
}

void World::setDirectionalLight(common::EntityId entityId,
                                const DirectionalLightComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setDirectionalLight requires valid entity id.");
        return;
    }

    if (!mImpl->requireAliveEntity(entityId, "setDirectionalLight"))
    {
        return;
    }

    std::uint32_t lightIndex = 0u;
    if (!mImpl->tryGetLightIndexForType(entityId, graphics::GpuLightType::Directional,
                                        "setDirectionalLight", lightIndex))
    {
        return;
    }

    if (lightIndex == kInvalidSlot)
    {
        const std::uint32_t envIndex  = entityEnvironment(entityId);
        const std::uint32_t lightSlot = mImpl->allocateLightSlot(envIndex, true);
        if (lightSlot == kInvalidSlot)
        {
            return;
        }
        lightIndex = envIndex * mImpl->mSceneLayout.maxLightsPerEnv + lightSlot;
        mImpl->mRenderLightIndices[entityId]       = lightIndex;
        mImpl->mRenderLights[lightIndex].entityId  = entityId;
        mImpl->mRenderLights[lightIndex].envIndex  = envIndex;
        mImpl->mRenderLights[lightIndex].lightSlot = lightSlot;
    }

    graphics::LightData &lightData     = mImpl->mRenderLights[lightIndex];
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
    mImpl->markLightDirty(lightIndex);
}

void World::setPointLight(common::EntityId entityId, const PointLightComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setPointLight requires valid entity id.");
        return;
    }

    if (!mImpl->requireAliveEntity(entityId, "setPointLight"))
    {
        return;
    }

    std::uint32_t lightIndex = 0u;
    if (!mImpl->tryGetLightIndexForType(entityId, graphics::GpuLightType::Point, "setPointLight",
                                        lightIndex))
    {
        return;
    }

    if (lightIndex == kInvalidSlot)
    {
        const std::uint32_t envIndex  = entityEnvironment(entityId);
        const std::uint32_t lightSlot = mImpl->allocateLightSlot(envIndex, false);
        if (lightSlot == kInvalidSlot)
        {
            return;
        }
        lightIndex = envIndex * mImpl->mSceneLayout.maxLightsPerEnv + lightSlot;
        mImpl->mRenderLightIndices[entityId]       = lightIndex;
        mImpl->mRenderLights[lightIndex].entityId  = entityId;
        mImpl->mRenderLights[lightIndex].envIndex  = envIndex;
        mImpl->mRenderLights[lightIndex].lightSlot = lightSlot;
    }

    const TransformComponent transform = tryGetTransform(entityId).value_or(TransformComponent{});
    graphics::LightData &lightData     = mImpl->mRenderLights[lightIndex];
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
    mImpl->markLightDirty(lightIndex);
}

void World::setSpotLight(common::EntityId entityId, const SpotLightComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setSpotLight requires valid entity id.");
        return;
    }

    if (!mImpl->requireAliveEntity(entityId, "setSpotLight"))
    {
        return;
    }

    std::uint32_t lightIndex = 0u;
    if (!mImpl->tryGetLightIndexForType(entityId, graphics::GpuLightType::Spot, "setSpotLight",
                                        lightIndex))
    {
        return;
    }

    if (lightIndex == kInvalidSlot)
    {
        const std::uint32_t envIndex  = entityEnvironment(entityId);
        const std::uint32_t lightSlot = mImpl->allocateLightSlot(envIndex, false);
        if (lightSlot == kInvalidSlot)
        {
            return;
        }
        lightIndex = envIndex * mImpl->mSceneLayout.maxLightsPerEnv + lightSlot;
        mImpl->mRenderLightIndices[entityId]       = lightIndex;
        mImpl->mRenderLights[lightIndex].entityId  = entityId;
        mImpl->mRenderLights[lightIndex].envIndex  = envIndex;
        mImpl->mRenderLights[lightIndex].lightSlot = lightSlot;
    }

    const TransformComponent transform = tryGetTransform(entityId).value_or(TransformComponent{});
    graphics::LightData &lightData     = mImpl->mRenderLights[lightIndex];
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
    mImpl->markLightDirty(lightIndex);
}

void World::setRigidBody(common::EntityId entityId, const RigidBodyComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setRigidBody requires valid entity id.");
        return;
    }

    if (!mImpl->requireAliveEntity(entityId, "setRigidBody"))
    {
        return;
    }

    mImpl->ensureEntityPoseSlot(entityId);

    TransformComponent transform{};
    if (const std::optional<TransformComponent> t = tryGetTransform(entityId))
    {
        transform = *t;
    }

    physics::RigidBodyState state{};
    state.entityId                    = entityId;
    state.position                    = transform.worldTransform.position;
    state.rotation                    = transform.worldTransform.rotation;
    state.scale                       = transform.worldTransform.scale;
    state.linearVelocity              = component.linearVelocity;
    state.angularVelocity             = component.angularVelocity;
    state.inverseMass                 = component.inverseMass;
    state.inverseInertiaLocal         = component.inverseInertiaLocal;
    state.proxyParticleLocalPositions = component.proxyParticleLocalPositions;
    state.proxyParticleMaterial       = component.proxyParticleMaterial;
    state.proxyParticleRadius         = component.proxyParticleRadius;
    state.proxyCollisionLayer         = component.proxyCollisionLayer;
    state.proxyCollisionMask          = component.proxyCollisionMask;
    state.suturingEnabled             = component.suturingEnabled;
    state.needleTipProxyIndex         = component.needleTipProxyIndex;
    state.bodyType                    = component.bodyType;
    state.environmentIndex            = entityEnvironment(entityId);
    state.kinematicTargetPosition     = component.kinematicTargetPosition;
    state.kinematicTargetRotation     = component.kinematicTargetRotation;
    state.kinematicTargetEnabled      = component.kinematicTargetEnabled;

    mImpl->mPhysicsWorld.upsertRigidBody(state);
    mImpl->mPhysicsLinks[entityId].hasRigidBody = true;
    mImpl->mPhysicsRenderableMappingsDirty      = true;
}

bool World::removeRigidBody(common::EntityId entityId)
{
    auto it = mImpl->mPhysicsLinks.find(entityId);
    if (it != mImpl->mPhysicsLinks.end())
    {
        it->second.hasRigidBody = false;
    }

    const bool removed = mImpl->mPhysicsWorld.removeRigidBody(entityId);
    if (removed)
    {
        mImpl->clearColliderLinks(entityId);
        mImpl->mPhysicsRenderableMappingsDirty = true;
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

    if (!mImpl->requireAliveEntity(entityId, "setSoftBody"))
    {
        return false;
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
    state.edgeFailureThreshold = component.edgeFailureThreshold;
    state.edgeCutResistance    = component.edgeCutResistance;
    state.volumeCompliance     = component.volumeCompliance;
    state.selfCollisionEnabled = component.selfCollisionEnabled;
    state.supportsSuturing     = component.supportsSuturing;
    state.collisionLayer       = component.collisionLayer;
    state.collisionMask        = component.collisionMask;

    if (!mImpl->mPhysicsWorld.upsertSoftBody(state))
    {
        return false;
    }
    (void)clearUltrasoundScattererAmplitudeRanges(entityId);
    mImpl->mPhysicsLinks[entityId].hasSoftBody = true;
    mImpl->mDrawRegistryDirty                  = true;
    mImpl->mSoftBodyRenderBindingsDirty        = true;
    return true;
}

bool World::setMeshfreeSoftBody(common::EntityId entityId,
                                const MeshfreeSoftBodyComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setMeshfreeSoftBody requires valid entity id.");
        return false;
    }

    if (!mImpl->requireAliveEntity(entityId, "setMeshfreeSoftBody"))
    {
        return false;
    }

    TransformComponent transform{};
    if (const std::optional<TransformComponent> t = tryGetTransform(entityId))
    {
        transform = *t;
    }

    physics::SoftBodyState state{};
    state.entityId         = entityId;
    state.environmentIndex = entityEnvironment(entityId);
    state.source.kind      = physics::SoftBodySourceKind::MeshfreeParticles;
    state.source.meshfreeParticles.particleRestPositions = component.particles;
    state.source.meshfreeParticles.surfaceRestPositions  = component.surfaceRestPositions;
    state.source.meshfreeParticles.surfaceNormals        = component.surfaceNormals;
    state.source.meshfreeParticles.surfaceTriangles      = component.surfaceTriangles;
    state.source.meshfreeParticles.staticParticleIndices = component.staticParticleIndices;
    state.source.meshfreeParticles.neighbourCount        = component.neighbourCount;
    state.material             = component.material;
    state.restTransform        = transform.worldTransform;
    state.particleMass         = component.particleMass;
    state.particleRadius       = component.particleRadius;
    state.edgeCompliance       = component.compliance;
    state.edgeFailureThreshold = component.edgeFailureThreshold;
    state.edgeCutResistance    = component.edgeCutResistance;
    state.volumeCompliance     = 0.0f;
    state.shapeMatching        = component.shapeMatching;
    state.selfCollisionEnabled = component.selfCollisionEnabled;
    state.collisionLayer       = component.collisionLayer;
    state.collisionMask        = component.collisionMask;

    if (!mImpl->mPhysicsWorld.upsertSoftBody(state))
    {
        return false;
    }
    mImpl->mPhysicsLinks[entityId].hasSoftBody = true;
    mImpl->mDrawRegistryDirty                  = true;
    mImpl->mSoftBodyRenderBindingsDirty        = true;
    return true;
}

bool World::removeSoftBody(common::EntityId entityId)
{
    const bool clearedAmplitudeRanges = clearUltrasoundScattererAmplitudeRanges(entityId);
    auto it                           = mImpl->mPhysicsLinks.find(entityId);
    if (it != mImpl->mPhysicsLinks.end())
    {
        it->second.hasSoftBody = false;
    }
    if (const auto renderIt = mImpl->mRenderableIndices.find(entityId);
        renderIt != mImpl->mRenderableIndices.end())
    {
        mImpl->markRenderablePoseDirty(static_cast<std::uint32_t>(renderIt->second));
        mImpl->markRenderableMetadataDirty(static_cast<std::uint32_t>(renderIt->second));
    }
    mImpl->mDrawRegistryDirty           = true;
    mImpl->mSoftBodyRenderBindingsDirty = true;
    return mImpl->mPhysicsWorld.removeSoftBody(entityId) || clearedAmplitudeRanges;
}

bool World::setStrand(common::EntityId entityId, const StrandComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setStrand requires valid entity id.");
        return false;
    }

    if (!mImpl->requireAliveEntity(entityId, "setStrand"))
    {
        return false;
    }

    physics::StrandState state{};
    state.entityId               = entityId;
    state.environmentIndex       = entityEnvironment(entityId);
    state.material               = component.material;
    state.restPositions          = component.restPositions;
    state.staticParticleIndices  = component.staticParticleIndices;
    state.particleMass           = component.particleMass;
    state.particleRadius         = component.particleRadius;
    state.stretchShearCompliance = component.stretchShearCompliance;
    state.bendCompliance         = component.bendCompliance;
    state.twistCompliance        = component.twistCompliance;
    state.distanceCompliance     = component.distanceCompliance;
    state.rootMaterialNormal     = component.rootMaterialNormal;
    state.selfCollisionEnabled   = component.selfCollisionEnabled;
    state.suturingEnabled        = component.suturingEnabled;
    state.pathNodeSpacing        = component.pathNodeSpacing;
    state.collisionLayer         = component.collisionLayer;
    state.collisionMask          = component.collisionMask;

    if (!mImpl->mPhysicsWorld.upsertStrand(state))
    {
        return false;
    }

    mImpl->mPhysicsLinks[entityId].hasStrand = true;
    return true;
}

bool World::removeStrand(common::EntityId entityId)
{
    auto it = mImpl->mPhysicsLinks.find(entityId);
    if (it != mImpl->mPhysicsLinks.end())
    {
        it->second.hasStrand = false;
    }
    return mImpl->mPhysicsWorld.removeStrand(entityId);
}

void World::setProceduralDeformableCurveRender(
    common::EntityId entityId, const ProceduralDeformableCurveRenderComponent &component)
{
    if (!mImpl->requireAliveEntity(entityId, "setProceduralDeformableCurveRender"))
    {
        return;
    }

    mImpl->mProceduralDeformableCurveRenders[entityId] = component;
    if (const auto renderIt = mImpl->mRenderableIndices.find(entityId);
        renderIt != mImpl->mRenderableIndices.end())
    {
        mImpl->markRenderableMetadataDirty(static_cast<std::uint32_t>(renderIt->second));
    }
    mImpl->mCurveRenderBindingsDirty = true;
    mImpl->mDrawRegistryDirty        = true;
}

bool World::removeProceduralDeformableCurveRender(common::EntityId entityId)
{
    const bool removed = mImpl->mProceduralDeformableCurveRenders.erase(entityId) > 0u;
    if (removed)
    {
        if (const auto renderIt = mImpl->mRenderableIndices.find(entityId);
            renderIt != mImpl->mRenderableIndices.end())
        {
            mImpl->markRenderableMetadataDirty(static_cast<std::uint32_t>(renderIt->second));
        }
        mImpl->mCurveRenderBindingsDirty = true;
        mImpl->mDrawRegistryDirty        = true;
    }
    return removed;
}

physics::AuthoredParticleSequenceState &World::upsertParticleSequence(
    const physics::AuthoredParticleSequenceState &state)
{
    mImpl->mCurveRenderBindingsDirty = true;
    mImpl->mDrawRegistryDirty        = true;
    return mImpl->mPhysicsWorld.upsertParticleSequence(state);
}

physics::AuthoredParticleDistanceConstraintState &World::upsertParticleDistanceConstraint(
    const physics::AuthoredParticleDistanceConstraintState &state)
{
    return mImpl->mPhysicsWorld.upsertParticleDistanceConstraint(state);
}

bool World::upsertRigidParticleAttachmentConstraint(
    const physics::AuthoredRigidParticleAttachmentConstraintState &state,
    physics::AuthoredRigidParticleAttachmentConstraintState *outAuthored)
{
    return mImpl->mPhysicsWorld.upsertRigidParticleAttachmentConstraint(state, outAuthored);
}

bool World::upsertStrandRigidAttachmentConstraint(
    const physics::AuthoredStrandRigidAttachmentConstraintState &state,
    physics::AuthoredStrandRigidAttachmentConstraintState *outAuthored)
{
    return mImpl->mPhysicsWorld.upsertStrandRigidAttachmentConstraint(state, outAuthored);
}

bool World::upsertRigidDistanceConstraint(
    const physics::AuthoredRigidDistanceConstraintState &state,
    physics::AuthoredRigidDistanceConstraintState *outAuthored)
{
    return mImpl->mPhysicsWorld.upsertRigidDistanceConstraint(state, outAuthored);
}

bool World::upsertRoutedCableConstraint(const physics::AuthoredRoutedCableConstraintState &state,
                                        physics::AuthoredRoutedCableConstraintState *outAuthored)
{
    return mImpl->mPhysicsWorld.upsertRoutedCableConstraint(state, outAuthored);
}

bool World::upsertBallJoint(const physics::BallJointState &state)
{
    if (!mImpl->requireAliveEntity(state.bodyA, "upsertBallJoint") ||
        !mImpl->requireAliveEntity(state.bodyB, "upsertBallJoint"))
    {
        return false;
    }
    const physics::RigidBodyState *bodyA = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyA);
    const physics::RigidBodyState *bodyB = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyB);
    if (bodyA == nullptr || bodyB == nullptr)
    {
        CRESSIM_LOG_ERROR("upsertBallJoint requires rigid bodies on both connected entities.");
        return false;
    }
    if (entityEnvironment(state.bodyA) != entityEnvironment(state.bodyB))
    {
        CRESSIM_LOG_ERROR("upsertBallJoint requires both connected rigid bodies to live in the "
                          "same environment.");
        return false;
    }
    physics::BallJointState resolved = state;
    resolved.bodyA                   = bodyA->rigidBodyId;
    resolved.bodyB                   = bodyB->rigidBodyId;
    return mImpl->mPhysicsWorld.upsertBallJoint(resolved);
}

bool World::upsertHingeJoint(const physics::HingeJointState &state)
{
    if (!mImpl->requireAliveEntity(state.bodyA, "upsertHingeJoint") ||
        !mImpl->requireAliveEntity(state.bodyB, "upsertHingeJoint"))
    {
        return false;
    }
    const physics::RigidBodyState *bodyA = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyA);
    const physics::RigidBodyState *bodyB = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyB);
    if (bodyA == nullptr || bodyB == nullptr)
    {
        CRESSIM_LOG_ERROR("upsertHingeJoint requires rigid bodies on both connected entities.");
        return false;
    }
    if (entityEnvironment(state.bodyA) != entityEnvironment(state.bodyB))
    {
        CRESSIM_LOG_ERROR("upsertHingeJoint requires both connected rigid bodies to live in the "
                          "same environment.");
        return false;
    }
    physics::HingeJointState resolved = state;
    resolved.bodyA                    = bodyA->rigidBodyId;
    resolved.bodyB                    = bodyB->rigidBodyId;
    return mImpl->mPhysicsWorld.upsertHingeJoint(resolved);
}

bool World::upsertSphericalJoint(const physics::SphericalJointState &state)
{
    if (!mImpl->requireAliveEntity(state.bodyA, "upsertSphericalJoint") ||
        !mImpl->requireAliveEntity(state.bodyB, "upsertSphericalJoint"))
    {
        return false;
    }
    const physics::RigidBodyState *bodyA = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyA);
    const physics::RigidBodyState *bodyB = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyB);
    if (bodyA == nullptr || bodyB == nullptr)
    {
        CRESSIM_LOG_ERROR("upsertSphericalJoint requires rigid bodies on both connected entities.");
        return false;
    }
    if (entityEnvironment(state.bodyA) != entityEnvironment(state.bodyB))
    {
        CRESSIM_LOG_ERROR("upsertSphericalJoint requires both connected rigid bodies to live in "
                          "the same environment.");
        return false;
    }
    physics::SphericalJointState resolved = state;
    resolved.bodyA                        = bodyA->rigidBodyId;
    resolved.bodyB                        = bodyB->rigidBodyId;
    return mImpl->mPhysicsWorld.upsertSphericalJoint(resolved);
}

bool World::upsertSliderJoint(const physics::SliderJointState &state)
{
    if (!mImpl->requireAliveEntity(state.bodyA, "upsertSliderJoint") ||
        !mImpl->requireAliveEntity(state.bodyB, "upsertSliderJoint"))
    {
        return false;
    }
    const physics::RigidBodyState *bodyA = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyA);
    const physics::RigidBodyState *bodyB = mImpl->mPhysicsWorld.tryGetRigidBody(state.bodyB);
    if (bodyA == nullptr || bodyB == nullptr)
    {
        CRESSIM_LOG_ERROR("upsertSliderJoint requires rigid bodies on both connected entities.");
        return false;
    }
    if (entityEnvironment(state.bodyA) != entityEnvironment(state.bodyB))
    {
        CRESSIM_LOG_ERROR("upsertSliderJoint requires both connected rigid bodies to live in the "
                          "same environment.");
        return false;
    }
    physics::SliderJointState resolved = state;
    resolved.bodyA                     = bodyA->rigidBodyId;
    resolved.bodyB                     = bodyB->rigidBodyId;
    return mImpl->mPhysicsWorld.upsertSliderJoint(resolved);
}

physics::AuthoredParticleCollisionFilterState &World::upsertParticleCollisionFilter(
    const physics::AuthoredParticleCollisionFilterState &state)
{
    return mImpl->mPhysicsWorld.upsertParticleCollisionFilter(state);
}

physics::AuthoredSuturingSequenceState &World::upsertSuturingSequence(
    const physics::AuthoredSuturingSequenceState &state)
{
    return mImpl->mPhysicsWorld.upsertSuturingSequence(state);
}

bool World::removeParticleDistanceConstraint(physics::ParticleConstraintId constraintId)
{
    return mImpl->mPhysicsWorld.removeParticleDistanceConstraint(constraintId);
}

bool World::removeRigidParticleAttachmentConstraint(
    physics::RigidParticleAttachmentConstraintId constraintId)
{
    return mImpl->mPhysicsWorld.removeRigidParticleAttachmentConstraint(constraintId);
}

bool World::removeStrandRigidAttachmentConstraint(
    physics::StrandRigidAttachmentConstraintId constraintId)
{
    return mImpl->mPhysicsWorld.removeStrandRigidAttachmentConstraint(constraintId);
}

bool World::removeRigidDistanceConstraint(physics::RigidDistanceConstraintId constraintId)
{
    return mImpl->mPhysicsWorld.removeRigidDistanceConstraint(constraintId);
}

bool World::removeRoutedCableConstraint(physics::RoutedCableConstraintId constraintId)
{
    return mImpl->mPhysicsWorld.removeRoutedCableConstraint(constraintId);
}

bool World::removeBallJoint(const physics::BallJointId jointId)
{
    return mImpl->mPhysicsWorld.removeBallJoint(jointId);
}

bool World::removeHingeJoint(const physics::HingeJointId jointId)
{
    return mImpl->mPhysicsWorld.removeHingeJoint(jointId);
}

bool World::removeSphericalJoint(const physics::SphericalJointId jointId)
{
    return mImpl->mPhysicsWorld.removeSphericalJoint(jointId);
}

bool World::removeSliderJoint(const physics::SliderJointId jointId)
{
    return mImpl->mPhysicsWorld.removeSliderJoint(jointId);
}

bool World::removeParticleCollisionFilter(physics::ParticleCollisionFilterId filterId)
{
    return mImpl->mPhysicsWorld.removeParticleCollisionFilter(filterId);
}

bool World::removeParticleSequence(physics::ParticleSequenceId sequenceId)
{
    const bool removed = mImpl->mPhysicsWorld.removeParticleSequence(sequenceId);
    if (removed)
    {
        mImpl->mCurveRenderBindingsDirty = true;
        mImpl->mDrawRegistryDirty        = true;
    }
    return removed;
}

bool World::removeSuturingSequence(physics::SuturingSequenceId sequenceId)
{
    return mImpl->mPhysicsWorld.removeSuturingSequence(sequenceId);
}

bool World::setFluid(common::EntityId entityId, const FluidComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setFluid requires valid entity id.");
        return false;
    }

    if (!mImpl->requireAliveEntity(entityId, "setFluid"))
    {
        return false;
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
    state.collisionLayer   = component.collisionLayer;
    state.collisionMask    = component.collisionMask;

    if (!mImpl->mPhysicsWorld.upsertFluid(state))
    {
        return false;
    }
    mImpl->mPhysicsLinks[entityId].hasFluid = true;
    mImpl->mDrawRegistryDirty               = true;
    return true;
}

bool World::removeFluid(common::EntityId entityId)
{
    auto it = mImpl->mPhysicsLinks.find(entityId);
    if (it != mImpl->mPhysicsLinks.end())
    {
        it->second.hasFluid = false;
    }
    mImpl->mDrawRegistryDirty = true;
    return mImpl->mPhysicsWorld.removeFluid(entityId);
}

void World::setUltrasoundProbe(common::EntityId entityId, const UltrasoundProbeComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setUltrasoundProbe requires valid entity id.");
        return;
    }
    if (!mImpl->requireAliveEntity(entityId, "setUltrasoundProbe"))
    {
        return;
    }

    if (!component.enabled)
    {
        (void)removeUltrasoundProbe(entityId);
        return;
    }

    mImpl->mUltrasoundProbes[entityId] = component;
}

bool World::removeUltrasoundProbe(common::EntityId entityId)
{
    clearUltrasoundProbeResult(entityId);
    return mImpl->mUltrasoundProbes.erase(entityId) > 0u;
}

void World::setUltrasoundRenderer(common::EntityId entityId,
                                  const UltrasoundRendererComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setUltrasoundRenderer requires valid entity id.");
        return;
    }
    if (!mImpl->requireAliveEntity(entityId, "setUltrasoundRenderer"))
    {
        return;
    }

    mImpl->mUltrasoundRenderers[entityId] = component;
}

bool World::removeUltrasoundRenderer(common::EntityId entityId)
{
    clearUltrasoundProbeResult(entityId);
    return mImpl->mUltrasoundRenderers.erase(entityId) > 0u;
}

void World::setUltrasoundScattererSource(common::EntityId entityId,
                                         const UltrasoundScattererSourceComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setUltrasoundScattererSource requires valid entity id.");
        return;
    }
    if (!mImpl->requireAliveEntity(entityId, "setUltrasoundScattererSource"))
    {
        return;
    }

    if (!component.enabled)
    {
        (void)removeUltrasoundScattererSource(entityId);
        return;
    }

    mImpl->mUltrasoundScattererSources[entityId] = component;
}

bool World::removeUltrasoundScattererSource(common::EntityId entityId)
{
    const bool removedSource = mImpl->mUltrasoundScattererSources.erase(entityId) > 0u;
    const bool removedRanges = clearUltrasoundScattererAmplitudeRanges(entityId);
    return removedSource || removedRanges;
}

void World::setUltrasoundScattererAmplitudeRanges(
    common::EntityId entityId, const std::vector<UltrasoundAmplitudeRange> &ranges)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("setUltrasoundScattererAmplitudeRanges requires valid entity id.");
        return;
    }
    if (!mImpl->requireAliveEntity(entityId, "setUltrasoundScattererAmplitudeRanges"))
    {
        return;
    }

    std::vector<Diligent::float3> authoredRestPositions;
    if (!mImpl->mPhysicsWorld.tryGetSoftBodyAuthoringRestPositions(entityId, authoredRestPositions))
    {
        CRESSIM_LOG_ERROR("setUltrasoundScattererAmplitudeRanges requires a soft body on entity ",
                          entityId, ".");
        return;
    }
    if (ranges.size() != authoredRestPositions.size())
    {
        CRESSIM_LOG_ERROR("setUltrasoundScattererAmplitudeRanges expected ",
                          authoredRestPositions.size(), " ranges for entity ", entityId, ", got ",
                          ranges.size(), ".");
        return;
    }

    mImpl->mUltrasoundScattererAmplitudeRanges[entityId] = ranges;
    ++mImpl->mUltrasoundScattererAmplitudeRevision;
}

bool World::clearUltrasoundScattererAmplitudeRanges(common::EntityId entityId)
{
    if (mImpl->mUltrasoundScattererAmplitudeRanges.erase(entityId) == 0u)
    {
        return false;
    }
    ++mImpl->mUltrasoundScattererAmplitudeRevision;
    return true;
}

World::ColliderHandle World::addCollider(common::EntityId entityId,
                                         const ColliderComponent &component)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("addCollider requires valid entity id.");
        return {};
    }

    if (!mImpl->requireAliveEntity(entityId, "addCollider"))
    {
        return {};
    }

    auto body = mImpl->mPhysicsLinks.find(entityId);
    if (body == mImpl->mPhysicsLinks.end() || !body->second.hasRigidBody)
    {
        CRESSIM_LOG_ERROR("addCollider requires a rigid body on the entity.");
        return {};
    }

    ColliderHandle handle{mImpl->mNextColliderId++};
    mImpl->mPhysicsWorld.upsertCollider(makeColliderState(entityId, handle.id, component));
    mImpl->mPhysicsLinks[entityId].colliders.push_back(handle);
    mImpl->mColliderOwnerEntity[handle.id] = entityId;
    return handle;
}

void World::updateCollider(ColliderHandle handle, const ColliderComponent &component)
{
    if (!handle.isValid())
    {
        CRESSIM_LOG_ERROR("updateCollider requires valid collider handle.");
        return;
    }

    const auto ownerIt = mImpl->mColliderOwnerEntity.find(handle.id);
    if (ownerIt == mImpl->mColliderOwnerEntity.end())
    {
        CRESSIM_LOG_ERROR("updateCollider received an unknown collider handle.");
        return;
    }

    const common::EntityId entityId = ownerIt->second;
    mImpl->mPhysicsWorld.upsertCollider(makeColliderState(entityId, handle.id, component));
}

bool World::removeCollider(ColliderHandle handle)
{
    if (!handle.isValid())
    {
        return false;
    }

    const auto ownerIt = mImpl->mColliderOwnerEntity.find(handle.id);
    if (ownerIt == mImpl->mColliderOwnerEntity.end())
    {
        return false;
    }

    const common::EntityId entityId = ownerIt->second;
    auto physIt                     = mImpl->mPhysicsLinks.find(entityId);
    if (physIt != mImpl->mPhysicsLinks.end())
    {
        auto &handles = physIt->second.colliders;
        handles.erase(std::remove_if(handles.begin(), handles.end(),
                                     [&](const ColliderHandle h) { return h.id == handle.id; }),
                      handles.end());
    }

    mImpl->mColliderOwnerEntity.erase(ownerIt);
    return mImpl->mPhysicsWorld.removeCollider(handle.id);
}

bool World::replaceColliders(common::EntityId entityId,
                             const std::vector<ColliderComponent> &components)
{
    if (entityId == common::kInvalidEntityId)
    {
        CRESSIM_LOG_ERROR("replaceColliders requires valid entity id.");
        return false;
    }
    if (!mImpl->requireAliveEntity(entityId, "replaceColliders"))
    {
        return false;
    }

    auto body = mImpl->mPhysicsLinks.find(entityId);
    if (body == mImpl->mPhysicsLinks.end() || !body->second.hasRigidBody)
    {
        CRESSIM_LOG_ERROR("replaceColliders requires a rigid body on the entity.");
        return false;
    }

    std::vector<physics::ColliderState> states;
    std::vector<ColliderHandle> handles;
    states.reserve(components.size());
    handles.reserve(components.size());
    for (const ColliderComponent &component : components)
    {
        const ColliderHandle handle{mImpl->mNextColliderId++};
        states.push_back(makeColliderState(entityId, handle.id, component));
        handles.push_back(handle);
    }

    mImpl->mPhysicsWorld.replaceColliders(entityId, states);
    mImpl->clearColliderLinks(entityId);
    body->second.colliders = handles;
    for (const ColliderHandle handle : handles)
    {
        mImpl->mColliderOwnerEntity[handle.id] = entityId;
    }
    return true;
}

bool World::removeTransform(common::EntityId entityId)
{
    const auto it = mImpl->mTransformIndex.find(entityId);
    if (it == mImpl->mTransformIndex.end())
    {
        return false;
    }

    const std::uint32_t index = it->second;
    const std::uint32_t last = static_cast<std::uint32_t>(mImpl->mTransforms.entityIds.size() - 1u);
    const common::EntityId movedEntity = mImpl->mTransforms.entityIds[last];

    if (index != last)
    {
        mImpl->mTransforms.entityIds[index]  = mImpl->mTransforms.entityIds[last];
        mImpl->mTransforms.components[index] = mImpl->mTransforms.components[last];
        mImpl->mTransformIndex[movedEntity]  = index;
    }

    mImpl->mTransforms.entityIds.pop_back();
    mImpl->mTransforms.components.pop_back();
    mImpl->mTransformIndex.erase(it);

    if (const auto it = mImpl->mRenderableIndices.find(entityId);
        it != mImpl->mRenderableIndices.end())
    {
        mImpl->markRenderablePoseDirty(static_cast<std::uint32_t>(it->second));
    }
    if (const auto it = mImpl->mRenderCameraIndices.find(entityId);
        it != mImpl->mRenderCameraIndices.end())
    {
        mImpl->markCameraDirty(static_cast<std::uint32_t>(it->second));
    }
    mImpl->markEntityPoseDirty(entityId);
    return true;
}

bool World::removeMeshRenderer(common::EntityId entityId)
{
    const auto it = mImpl->mRenderableIndices.find(entityId);
    if (it == mImpl->mRenderableIndices.end())
    {
        return false;
    }

    const std::uint32_t objectIndex              = static_cast<std::uint32_t>(it->second);
    const graphics::RenderableInstance &instance = mImpl->mRenderables[objectIndex];
    reclaimDenseSlot(mImpl->mFreeRenderableSlotsByEnv, instance.envIndex, instance.objectSlot);
    mImpl->mRenderables[objectIndex] = {};
    mImpl->markRenderablePoseDirty(objectIndex);
    mImpl->markRenderableMetadataDirty(objectIndex);
    mImpl->mRenderableIndices.erase(it);
    mImpl->mDrawRegistryDirty              = true;
    mImpl->mPhysicsRenderableMappingsDirty = true;
    mImpl->mSoftBodyRenderBindingsDirty    = true;
    mImpl->mCurveRenderBindingsDirty       = true;
    return true;
}

bool World::removeCamera(common::EntityId entityId)
{
    const auto it = mImpl->mRenderCameraIndices.find(entityId);
    if (it == mImpl->mRenderCameraIndices.end())
    {
        return false;
    }

    const std::uint32_t cameraIndex    = static_cast<std::uint32_t>(it->second);
    const graphics::CameraData &camera = mImpl->mRenderCameras[cameraIndex];
    reclaimDenseSlot(mImpl->mFreeCameraSlotsByEnv, camera.envIndex, camera.cameraSlot);
    mImpl->mRenderCameras[cameraIndex] = {};
    mImpl->mRenderCameraIndices.erase(it);
    mImpl->markCameraDirty(cameraIndex);
    return true;
}

bool World::removeDirectionalLight(common::EntityId entityId)
{
    const auto component = tryGetDirectionalLight(entityId);
    if (!component.has_value())
    {
        return false;
    }
    return mImpl->removeLight(entityId);
}

bool World::removePointLight(common::EntityId entityId)
{
    const auto component = tryGetPointLight(entityId);
    if (!component.has_value())
    {
        return false;
    }
    return mImpl->removeLight(entityId);
}

bool World::removeSpotLight(common::EntityId entityId)
{
    const auto component = tryGetSpotLight(entityId);
    if (!component.has_value())
    {
        return false;
    }
    return mImpl->removeLight(entityId);
}

std::optional<TransformComponent> World::tryGetTransform(common::EntityId entityId) const
{
    const auto it = mImpl->mTransformIndex.find(entityId);
    if (it == mImpl->mTransformIndex.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = it->second;
    return mImpl->mTransforms.components[index];
}

std::optional<MeshRendererComponent> World::tryGetMeshRenderer(common::EntityId entityId) const
{
    const auto it = mImpl->mRenderableIndices.find(entityId);
    if (it == mImpl->mRenderableIndices.end())
    {
        return std::nullopt;
    }

    const std::uint32_t index = static_cast<std::uint32_t>(it->second);
    MeshRendererComponent component{};
    component.mesh           = mImpl->mRenderables[index].mesh;
    component.material       = mImpl->mRenderables[index].material;
    component.segmentationId = mImpl->mRenderables[index].segmentationId;
    component.visible        = mImpl->mRenderables[index].visible;
    return component;
}

std::optional<CameraComponent> World::tryGetCamera(common::EntityId entityId) const
{
    const auto it = mImpl->mRenderCameraIndices.find(entityId);
    if (it == mImpl->mRenderCameraIndices.end())
    {
        return std::nullopt;
    }

    const graphics::CameraData &camera =
        mImpl->mRenderCameras[static_cast<std::uint32_t>(it->second)];
    CameraComponent component{};
    component.verticalFovDegrees = camera.verticalFovDegrees;
    component.nearClip           = camera.nearClip;
    component.farClip            = camera.farClip;
    switch (camera.product)
    {
    case graphics::CameraData::Product::Depth:
        component.product = CameraComponent::Product::Depth;
        break;
    case graphics::CameraData::Product::SegmentationDepth:
        component.product = CameraComponent::Product::SegmentationDepth;
        break;
    case graphics::CameraData::Product::ColorDepth:
    default:
        component.product = CameraComponent::Product::ColorDepth;
        break;
    }
    component.output          = camera.output;
    component.outputWidth     = camera.outputWidth;
    component.outputHeight    = camera.outputHeight;
    component.viewport        = camera.viewport;
    component.clearColor      = camera.clearColor;
    component.clearDepth      = camera.clearDepth;
    component.clearColorValue = camera.clearColorValue;
    component.clearDepthValue = camera.clearDepthValue;
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
    const auto it = mImpl->mRenderLightIndices.find(entityId);
    if (it == mImpl->mRenderLightIndices.end())
    {
        return std::nullopt;
    }

    const graphics::LightData &light = mImpl->mRenderLights[static_cast<std::uint32_t>(it->second)];
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
    const auto it = mImpl->mRenderLightIndices.find(entityId);
    if (it == mImpl->mRenderLightIndices.end())
    {
        return std::nullopt;
    }

    const graphics::LightData &light = mImpl->mRenderLights[static_cast<std::uint32_t>(it->second)];
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
    const auto it = mImpl->mRenderLightIndices.find(entityId);
    if (it == mImpl->mRenderLightIndices.end())
    {
        return std::nullopt;
    }

    const graphics::LightData &light = mImpl->mRenderLights[static_cast<std::uint32_t>(it->second)];
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
    const physics::RigidBodyState *rb = mImpl->mPhysicsWorld.tryGetRigidBody(entityId);
    if (!rb)
    {
        return std::nullopt;
    }

    RigidBodyComponent component{};
    component.bodyType                    = rb->bodyType;
    component.linearVelocity              = rb->linearVelocity;
    component.angularVelocity             = rb->angularVelocity;
    component.inverseMass                 = rb->inverseMass;
    component.inverseInertiaLocal         = rb->inverseInertiaLocal;
    component.proxyParticleLocalPositions = rb->proxyParticleLocalPositions;
    component.proxyParticleMaterial       = rb->proxyParticleMaterial;
    component.proxyParticleRadius         = rb->proxyParticleRadius;
    component.proxyCollisionLayer         = rb->proxyCollisionLayer;
    component.proxyCollisionMask          = rb->proxyCollisionMask;
    component.suturingEnabled             = rb->suturingEnabled;
    component.needleTipProxyIndex         = rb->needleTipProxyIndex;
    component.kinematicTargetPosition     = rb->kinematicTargetPosition;
    component.kinematicTargetRotation     = rb->kinematicTargetRotation;
    component.kinematicTargetEnabled      = rb->kinematicTargetEnabled;
    return component;
}

std::optional<SoftBodyComponent> World::tryGetSoftBody(common::EntityId entityId) const
{
    const physics::SoftBodyState *softBody = mImpl->mPhysicsWorld.tryGetSoftBody(entityId);
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
    component.edgeFailureThreshold = softBody->edgeFailureThreshold;
    component.edgeCutResistance    = softBody->edgeCutResistance;
    component.volumeCompliance     = softBody->volumeCompliance;
    component.selfCollisionEnabled = softBody->selfCollisionEnabled;
    component.supportsSuturing     = softBody->supportsSuturing;
    component.collisionLayer       = softBody->collisionLayer;
    component.collisionMask        = softBody->collisionMask;
    return component;
}

std::optional<StrandComponent> World::tryGetStrand(common::EntityId entityId) const
{
    const physics::StrandState *strand = mImpl->mPhysicsWorld.tryGetStrand(entityId);
    if (!strand)
    {
        return std::nullopt;
    }

    StrandComponent component{};
    component.material               = strand->material;
    component.restPositions          = strand->restPositions;
    component.staticParticleIndices  = strand->staticParticleIndices;
    component.particleMass           = strand->particleMass;
    component.particleRadius         = strand->particleRadius;
    component.stretchShearCompliance = strand->stretchShearCompliance;
    component.bendCompliance         = strand->bendCompliance;
    component.twistCompliance        = strand->twistCompliance;
    component.distanceCompliance     = strand->distanceCompliance;
    component.rootMaterialNormal     = strand->rootMaterialNormal;
    component.selfCollisionEnabled   = strand->selfCollisionEnabled;
    component.suturingEnabled        = strand->suturingEnabled;
    component.pathNodeSpacing        = strand->pathNodeSpacing;
    component.collisionLayer         = strand->collisionLayer;
    component.collisionMask          = strand->collisionMask;
    return component;
}

std::optional<ProceduralDeformableCurveRenderComponent> World::
    tryGetProceduralDeformableCurveRender(common::EntityId entityId) const
{
    const auto it = mImpl->mProceduralDeformableCurveRenders.find(entityId);
    return it != mImpl->mProceduralDeformableCurveRenders.end()
               ? std::optional<ProceduralDeformableCurveRenderComponent>{it->second}
               : std::nullopt;
}

std::optional<FluidComponent> World::tryGetFluid(common::EntityId entityId) const
{
    const physics::FluidState *fluid = mImpl->mPhysicsWorld.tryGetFluid(entityId);
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
    component.collisionLayer = fluid->collisionLayer;
    component.collisionMask  = fluid->collisionMask;
    return component;
}

const physics::AuthoredParticleDistanceConstraintState *World::tryGetParticleDistanceConstraint(
    physics::ParticleConstraintId constraintId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetParticleDistanceConstraint(constraintId);
}

const physics::AuthoredRigidParticleAttachmentConstraintState *World::
    tryGetRigidParticleAttachmentConstraint(
        physics::RigidParticleAttachmentConstraintId constraintId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetRigidParticleAttachmentConstraint(constraintId);
}

const physics::AuthoredStrandRigidAttachmentConstraintState *World::
    tryGetStrandRigidAttachmentConstraint(
        physics::StrandRigidAttachmentConstraintId constraintId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetStrandRigidAttachmentConstraint(constraintId);
}

const physics::AuthoredRigidDistanceConstraintState *World::tryGetRigidDistanceConstraint(
    physics::RigidDistanceConstraintId constraintId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetRigidDistanceConstraint(constraintId);
}

const physics::AuthoredRoutedCableConstraintState *World::tryGetRoutedCableConstraint(
    physics::RoutedCableConstraintId constraintId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetRoutedCableConstraint(constraintId);
}

const physics::BallJointState *World::tryGetBallJoint(
    const physics::BallJointId jointId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetBallJoint(jointId);
}

const physics::HingeJointState *World::tryGetHingeJoint(
    const physics::HingeJointId jointId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetHingeJoint(jointId);
}

const physics::SphericalJointState *World::tryGetSphericalJoint(
    const physics::SphericalJointId jointId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetSphericalJoint(jointId);
}

const physics::SliderJointState *World::tryGetSliderJoint(
    const physics::SliderJointId jointId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetSliderJoint(jointId);
}

const physics::AuthoredParticleCollisionFilterState *World::tryGetParticleCollisionFilter(
    physics::ParticleCollisionFilterId filterId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetParticleCollisionFilter(filterId);
}

const physics::AuthoredParticleSequenceState *World::tryGetParticleSequence(
    physics::ParticleSequenceId sequenceId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetParticleSequence(sequenceId);
}

const physics::AuthoredSuturingSequenceState *World::tryGetSuturingSequence(
    physics::SuturingSequenceId sequenceId) const noexcept
{
    return mImpl->mPhysicsWorld.tryGetSuturingSequence(sequenceId);
}

std::optional<UltrasoundProbeComponent> World::tryGetUltrasoundProbe(
    common::EntityId entityId) const
{
    const auto it = mImpl->mUltrasoundProbes.find(entityId);
    return it != mImpl->mUltrasoundProbes.end()
               ? std::optional<UltrasoundProbeComponent>{it->second}
               : std::nullopt;
}

std::optional<UltrasoundRendererComponent> World::tryGetUltrasoundRenderer(
    common::EntityId entityId) const
{
    const auto it = mImpl->mUltrasoundRenderers.find(entityId);
    return it != mImpl->mUltrasoundRenderers.end()
               ? std::optional<UltrasoundRendererComponent>{it->second}
               : std::nullopt;
}

std::optional<UltrasoundScattererSourceComponent> World::tryGetUltrasoundScattererSource(
    common::EntityId entityId) const
{
    const auto it = mImpl->mUltrasoundScattererSources.find(entityId);
    return it != mImpl->mUltrasoundScattererSources.end()
               ? std::optional<UltrasoundScattererSourceComponent>{it->second}
               : std::nullopt;
}

std::optional<SoftBodyAuthoringParticles> World::tryGetSoftBodyAuthoringParticles(
    common::EntityId entityId) const
{
    const physics::SoftBodyState *softBody = mImpl->mPhysicsWorld.tryGetSoftBody(entityId);
    if (softBody == nullptr)
    {
        return std::nullopt;
    }

    SoftBodyAuthoringParticles particles{};
    if (!mImpl->mPhysicsWorld.tryGetSoftBodyAuthoringRestPositions(entityId,
                                                                   particles.restPositions))
    {
        return std::nullopt;
    }
    particles.particleCount = static_cast<std::uint32_t>(particles.restPositions.size());
    return particles;
}

const std::vector<UltrasoundAmplitudeRange> *World::tryGetUltrasoundScattererAmplitudeRanges(
    common::EntityId entityId) const noexcept
{
    const auto it = mImpl->mUltrasoundScattererAmplitudeRanges.find(entityId);
    return it != mImpl->mUltrasoundScattererAmplitudeRanges.end() ? &it->second : nullptr;
}

const UltrasoundProbeResult *World::tryGetUltrasoundProbeResult(
    common::EntityId entityId) const noexcept
{
    const auto it = mImpl->mUltrasoundProbeResults.find(entityId);
    return it != mImpl->mUltrasoundProbeResults.end() ? &it->second : nullptr;
}

std::optional<ColliderComponent> World::tryGetCollider(ColliderHandle handle) const
{
    if (!handle.isValid())
    {
        return std::nullopt;
    }
    if (mImpl->mColliderOwnerEntity.find(handle.id) == mImpl->mColliderOwnerEntity.end())
    {
        return std::nullopt;
    }

    const physics::ColliderState *c = mImpl->mPhysicsWorld.tryGetCollider(handle.id);
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
    const auto it = mImpl->mPhysicsLinks.find(entityId);
    return it != mImpl->mPhysicsLinks.end() ? it->second.colliders : emptyColliderHandleList();
}

physics::PhysicsWorld &World::physicsWorld() noexcept
{
    return mImpl->mPhysicsWorld;
}

const physics::PhysicsWorld &World::physicsWorld() const noexcept
{
    return mImpl->mPhysicsWorld;
}

void World::setGpuEntityScene(const graphics::GpuEntitySceneView &sceneView) noexcept
{
    mImpl->mGpuEntityScene = sceneView;
}

const std::vector<graphics::RenderableInstance> &World::renderables() const noexcept
{
    return mImpl->mRenderables;
}

const std::vector<graphics::CameraData> &World::cameras() const noexcept
{
    return mImpl->mRenderCameras;
}

const std::vector<graphics::LightData> &World::lights() const noexcept
{
    return mImpl->mRenderLights;
}

std::uint32_t World::entityPoseSlot(common::EntityId entityId) const noexcept
{
    const auto it = mImpl->mEntityPoseSlotByEntity.find(entityId);
    return it != mImpl->mEntityPoseSlotByEntity.end() ? it->second : kInvalidSlot;
}

const std::vector<Diligent::float4> &World::entityPosePositions() const noexcept
{
    return mImpl->mEntityPosePositionsHost;
}

const std::vector<Diligent::float4> &World::entityPoseOrientations() const noexcept
{
    return mImpl->mEntityPoseOrientationsHost;
}

const std::vector<Diligent::float4> &World::entityPoseScales() const noexcept
{
    return mImpl->mEntityPoseScalesHost;
}

const std::vector<Diligent::float4> &World::renderObjectPositions() const noexcept
{
    return mImpl->mRenderObjectPositions;
}

const std::vector<Diligent::float4> &World::renderObjectOrientations() const noexcept
{
    return mImpl->mRenderObjectOrientations;
}

const std::vector<Diligent::float4> &World::renderObjectScales() const noexcept
{
    return mImpl->mRenderObjectScales;
}

const std::vector<graphics::GpuRenderableMetadata> &World::renderableMetadata() const noexcept
{
    return mImpl->mRenderableMetadataHost;
}

const std::vector<graphics::GpuRenderableQueueInfo> &World::renderableQueueInfo() const noexcept
{
    return mImpl->mRenderableQueueInfoHost;
}

const std::vector<graphics::GpuCameraInput> &World::cameraInputs() const noexcept
{
    return mImpl->mCameraInputsHost;
}

const std::vector<graphics::GpuLightInput> &World::lightInputs() const noexcept
{
    return mImpl->mLightInputsHost;
}

const std::vector<graphics::GpuLocalLightSelection> &World::localLightSelections() const noexcept
{
    return mImpl->mLocalLightSelectionsHost;
}

const std::vector<graphics::GpuSoftBodyVertexBinding> &World::softBodyVertexBindings()
    const noexcept
{
    return mImpl->mSoftBodyVertexBindingsHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry> &World::opaqueDrawRegistry()
    const noexcept
{
    return mImpl->mOpaqueDrawRegistryHost;
}

const std::vector<graphics::TransparentDrawEntry> &World::transparentDrawRegistry() const noexcept
{
    return mImpl->mTransparentDrawRegistryHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry> &World::shadowDrawRegistry()
    const noexcept
{
    return mImpl->mShadowDrawRegistryHost;
}

const std::vector<graphics::IndirectCommandRegistryEntry> &World::localShadowDrawRegistry()
    const noexcept
{
    return mImpl->mLocalShadowDrawRegistryHost;
}

const std::vector<EntityPoseMappingEntry> &World::physicsRenderableMappings()
{
    const std::uint64_t rigidBodyTopologyRevision =
        mImpl->mPhysicsWorld.rigidBodyTopologyRevision();
    if (!mImpl->mPhysicsRenderableMappingsDirty &&
        mImpl->mCachedPhysicsRenderableMappingsBodyTopologyRevision == rigidBodyTopologyRevision)
    {
        return mImpl->mPhysicsRenderableMappingsCache;
    }

    mImpl->mPhysicsRenderableMappingsCache.clear();

    const auto &rigidBodies = mImpl->mPhysicsWorld.rigidBodySoA();
    if (!rigidBodies.entityIds.empty())
    {
        for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(rigidBodies.entityIds.size()); ++i)
        {
            const common::EntityId entityId = rigidBodies.entityIds[i];
            const auto poseSlotIt           = mImpl->mEntityPoseSlotByEntity.find(entityId);
            if (poseSlotIt == mImpl->mEntityPoseSlotByEntity.end())
            {
                continue;
            }

            EntityPoseMappingEntry entry{};
            entry.sourcePoseIndex = i;
            entry.entityPoseIndex = poseSlotIt->second;
            mImpl->mPhysicsRenderableMappingsCache.push_back(entry);
        }
    }

    mImpl->mPhysicsRenderableMappingsDirty                      = false;
    mImpl->mCachedPhysicsRenderableMappingsBodyTopologyRevision = rigidBodyTopologyRevision;
    return mImpl->mPhysicsRenderableMappingsCache;
}

std::uint64_t World::entityPoseRevision() const noexcept
{
    return mImpl->mEntityPoseRevision;
}

std::uint64_t World::renderableMetadataRevision() const noexcept
{
    return mImpl->mRenderableMetadataRevision;
}

std::uint64_t World::renderableQueueInfoRevision() const noexcept
{
    return mImpl->mRenderableQueueInfoRevision;
}

std::uint64_t World::softBodyVertexBindingRevision() const noexcept
{
    return mImpl->mSoftBodyVertexBindingRevision;
}

std::uint64_t World::cameraInputRevision() const noexcept
{
    return mImpl->mCameraInputRevision;
}

std::uint64_t World::lightInputRevision() const noexcept
{
    return mImpl->mLightInputRevision;
}

std::uint64_t World::localLightSelectionRevision() const noexcept
{
    return mImpl->mLocalLightSelectionRevision;
}

const graphics::GpuEntitySceneView &World::gpuEntityScene() const noexcept
{
    return mImpl->mGpuEntityScene;
}

graphics::HostSceneView World::hostSceneView() const noexcept
{
    return graphics::HostSceneView{
        &mImpl->mRenderables,
        &mImpl->mRenderableMetadataHost,
        &mImpl->mRenderCameras,
        &mImpl->mRenderLights,
        &mImpl->mEnvironmentIbls,
        &mImpl->mEnvironmentFluids,
        &mImpl->mOpaqueDrawRegistryHost,
        &mImpl->mTransparentDrawRegistryHost,
        &mImpl->mShadowDrawRegistryHost,
        &mImpl->mLocalShadowDrawRegistryHost,
        &mImpl->mGpuEntityScene,
    };
}

void World::ensureRenderStateUpToDate(const graphics::RenderResourceManager &resources)
{
    const std::uint64_t softBodyTopologyRevision = mImpl->mPhysicsWorld.softBodyTopologyRevision();
    if (mImpl->mCachedSoftBodyRenderTopologyRevision != softBodyTopologyRevision)
    {
        mImpl->mSoftBodyRenderBindingsDirty          = true;
        mImpl->mCachedSoftBodyRenderTopologyRevision = softBodyTopologyRevision;
    }

    const std::uint64_t physicsRevision = mImpl->mPhysicsWorld.authoredRevision();
    if (mImpl->mCachedSoftBodyPhysicsRevision != physicsRevision)
    {
        for (std::uint32_t objectIndex = 0u;
             objectIndex < static_cast<std::uint32_t>(mImpl->mRenderables.size()); ++objectIndex)
        {
            const graphics::RenderableInstance &renderable = mImpl->mRenderables[objectIndex];
            if (renderable.entityId == common::kInvalidEntityId ||
                renderable.objectSlot == kInvalidSlot)
            {
                continue;
            }
            const auto physIt = mImpl->mPhysicsLinks.find(renderable.entityId);
            if (physIt != mImpl->mPhysicsLinks.end() && physIt->second.hasSoftBody)
            {
                mImpl->markRenderablePoseDirty(objectIndex);
                mImpl->markRenderableMetadataDirty(objectIndex);
            }
        }
        mImpl->mCachedSoftBodyPhysicsRevision = physicsRevision;
    }
    if (mImpl->mCachedCurveRenderPhysicsRevision != physicsRevision)
    {
        mImpl->mCurveRenderBindingsDirty         = true;
        mImpl->mCachedCurveRenderPhysicsRevision = physicsRevision;
    }

    if (mImpl->mSoftBodyRenderBindingsDirty)
    {
        mImpl->mPhysicsWorld.ensureSoftBodyDerivedStateUpToDate();
        mImpl->rebuildSoftBodyRenderBindings(resources);
    }
    if (mImpl->mCurveRenderBindingsDirty)
    {
        mImpl->mPhysicsWorld.ensureSoftBodyDerivedStateUpToDate();
        mImpl->rebuildCurveRenderBindings(resources);
    }

    for (const std::uint32_t entityPoseSlot : mImpl->mDirtyEntityPoseSlots)
    {
        mImpl->refreshEntityPoseSlot(entityPoseSlot);
    }

    for (const std::uint32_t objectIndex : mImpl->mDirtyRenderablePoseIndices)
    {
        mImpl->refreshRenderablePose(objectIndex);
    }

    for (const std::uint32_t cameraIndex : mImpl->mDirtyCameraIndices)
    {
        mImpl->refreshCameraEntry(cameraIndex);
    }

    for (const std::uint32_t lightIndex : mImpl->mDirtyLightIndices)
    {
        mImpl->refreshLightEntry(lightIndex);
    }

    if (!mImpl->mDirtyLightIndices.empty())
    {
        mImpl->rebuildLocalLightSelections();
    }

    mImpl->refreshDirtyRenderableMetadata(resources);
    if (mImpl->mDrawRegistryDirty)
    {
        mImpl->rebuildDrawRegistries(resources);
        mImpl->mDrawRegistryDirty = false;
    }

    mImpl->clearDirtyIndexSet(mImpl->mDirtyEntityPoseSlots, mImpl->mDirtyEntityPoseBits);
    mImpl->clearDirtyIndexSet(mImpl->mDirtyRenderablePoseIndices, mImpl->mDirtyRenderablePoseBits);
    mImpl->clearDirtyIndexSet(mImpl->mDirtyRenderableMetadataIndices,
                              mImpl->mDirtyRenderableMetadataBits);
    mImpl->clearDirtyIndexSet(mImpl->mDirtyCameraIndices, mImpl->mDirtyCameraBits);
    mImpl->clearDirtyIndexSet(mImpl->mDirtyLightIndices, mImpl->mDirtyLightBits);
}

bool World::setRenderableMeshResource(common::EntityId entityId, graphics::MeshHandle mesh)
{
    const auto indexIt = mImpl->mRenderableIndices.find(entityId);
    if (indexIt == mImpl->mRenderableIndices.end())
    {
        return false;
    }

    const std::uint32_t objectIndex = static_cast<std::uint32_t>(indexIt->second);
    if (objectIndex >= mImpl->mRenderables.size())
    {
        return false;
    }

    graphics::RenderableInstance &renderable = mImpl->mRenderables[objectIndex];
    if (renderable.mesh.id == mesh.id)
    {
        return true;
    }

    renderable.mesh = mesh;
    mImpl->markRenderableMetadataDirty(objectIndex);
    mImpl->mDrawRegistryDirty = true;
    return true;
}

const std::unordered_map<common::EntityId, UltrasoundProbeComponent> &World::
    ultrasoundProbeComponents() const noexcept
{
    return mImpl->mUltrasoundProbes;
}

const std::unordered_map<common::EntityId, UltrasoundRendererComponent> &World::
    ultrasoundRendererComponents() const noexcept
{
    return mImpl->mUltrasoundRenderers;
}

const std::unordered_map<common::EntityId, UltrasoundScattererSourceComponent> &World::
    ultrasoundScattererSourceComponents() const noexcept
{
    return mImpl->mUltrasoundScattererSources;
}

std::uint64_t World::ultrasoundScattererAmplitudeRevision() const noexcept
{
    return mImpl->mUltrasoundScattererAmplitudeRevision;
}

void World::setUltrasoundProbeResult(common::EntityId entityId, const UltrasoundProbeResult &result)
{
    mImpl->mUltrasoundProbeResults[entityId] = result;
}

void World::clearUltrasoundProbeResult(common::EntityId entityId)
{
    mImpl->mUltrasoundProbeResults.erase(entityId);
}

void World::Impl::rebuildSoftBodyRenderBindings(const graphics::RenderResourceManager &resources)
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

        const bool useNearestParticleSkinning =
            softBody->source.kind == physics::SoftBodySourceKind::MeshfreeParticles;
        std::unordered_map<QuantizedPointKey, std::uint32_t, QuantizedPointKeyHash>
            restIndexByQuantizedPosition;
        if (!useNearestParticleSkinning)
        {
            restIndexByQuantizedPosition.reserve(softBody->restPositions.size());
        }
        std::vector<Diligent::float3> restPositionsLocal;
        restPositionsLocal.reserve(softBody->restPositions.size());
        for (std::uint32_t localParticleIndex = 0u;
             localParticleIndex < static_cast<std::uint32_t>(softBody->restPositions.size());
             ++localParticleIndex)
        {
            const Diligent::float3 localRestPosition = inverseTransformPoint(
                softBody->restTransform, softBody->restPositions[localParticleIndex]);
            restPositionsLocal.push_back(localRestPosition);
            if (!useNearestParticleSkinning)
            {
                restIndexByQuantizedPosition.try_emplace(
                    quantizePoint(localRestPosition, kSoftBodyVertexMatchEpsilon),
                    localParticleIndex);
            }
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
            graphics::GpuSoftBodyVertexBinding binding{};
            if (useNearestParticleSkinning)
            {
                binding = makeNearestParticleSkinBinding(vertex.position, restPositionsLocal,
                                                         softBody->particleOffset);
            }
            else
            {
                const std::optional<std::uint32_t> matchedRestIndex = findMatchingRestVertexLocal(
                    vertex.position, restPositionsLocal, restIndexByQuantizedPosition,
                    kSoftBodyVertexMatchEpsilon);
                if (!matchedRestIndex.has_value())
                {
                    valid = false;
                    break;
                }

                binding = makeExactSoftBodyVertexBinding(softBody->particleOffset +
                                                         matchedRestIndex.value());
            }

            mSoftBodyVertexBindingsHost.push_back(binding);
            softRenderData.vertexBindings.push_back(
                physics::SoftRenderVertexBinding{binding.particleIndices, binding.weights});
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
                              useNearestParticleSkinning
                                  ? ": visual mesh vertices could not be bound to meshfree "
                                    "particles."
                                  : ": visual mesh vertices must match tet rest vertices exactly.");
            mSoftBodyVertexBindingsHost.resize(bindingBase);
            softRenderData.vertexBindings.resize(bindingBase);
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
                dominantParticleIndex(mSoftBodyVertexBindingsHost[bindingBase + i0]);
            const std::uint32_t particleIndex1 =
                dominantParticleIndex(mSoftBodyVertexBindingsHost[bindingBase + i1]);
            const std::uint32_t particleIndex2 =
                dominantParticleIndex(mSoftBodyVertexBindingsHost[bindingBase + i2]);
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
    bumpGeneration(mSoftBodyVertexBindingRevision);
}

void World::Impl::rebuildCurveRenderBindings(const graphics::RenderResourceManager &resources)
{
    std::fill(mCurveRenderVertexBaseByObject.begin(), mCurveRenderVertexBaseByObject.end(),
              kInvalidSlot);
    std::fill(mCurveRenderVertexNormalBaseByObject.begin(),
              mCurveRenderVertexNormalBaseByObject.end(), kInvalidSlot);
    std::fill(mCurveRenderIndexByObject.begin(), mCurveRenderIndexByObject.end(), kInvalidSlot);
    std::fill(mCurveRenderVertexCountByObject.begin(), mCurveRenderVertexCountByObject.end(), 0u);

    physics::CurveRenderDataHost curveRenderData;
    const auto resolveParticleReference =
        [this](const physics::AuthoredParticleReference &reference) -> std::optional<std::uint32_t>
    {
        switch (reference.type)
        {
        case physics::AuthoredParticleReferenceType::StrandParticle:
        {
            const physics::StrandState *strand = mPhysicsWorld.tryGetStrand(reference.entityId);
            if (strand == nullptr || reference.localParticleIndex >= strand->particleCount)
            {
                return std::nullopt;
            }
            return strand->particleOffset + reference.localParticleIndex;
        }
        case physics::AuthoredParticleReferenceType::SoftBodyParticle:
        {
            const physics::SoftBodyState *softBody =
                mPhysicsWorld.tryGetSoftBody(reference.entityId);
            if (softBody == nullptr || reference.localParticleIndex >= softBody->particleCount)
            {
                return std::nullopt;
            }
            return softBody->particleOffset + reference.localParticleIndex;
        }
        case physics::AuthoredParticleReferenceType::RigidProxyParticle:
            return std::nullopt;
        }
        return std::nullopt;
    };

    for (std::uint32_t objectIndex = 0u;
         objectIndex < static_cast<std::uint32_t>(mRenderables.size()); ++objectIndex)
    {
        const graphics::RenderableInstance &renderable = mRenderables[objectIndex];
        if (renderable.entityId == common::kInvalidEntityId ||
            renderable.objectSlot == kInvalidSlot)
        {
            continue;
        }

        const auto componentIt = mProceduralDeformableCurveRenders.find(renderable.entityId);
        if (componentIt == mProceduralDeformableCurveRenders.end())
        {
            continue;
        }

        const ProceduralDeformableCurveRenderComponent &component = componentIt->second;
        if (!component.enabled || component.sequenceId == physics::kInvalidParticleSequenceId)
        {
            continue;
        }

        const auto physicsLinkIt = mPhysicsLinks.find(renderable.entityId);
        const bool entityHasSoftBody =
            physicsLinkIt != mPhysicsLinks.end() && physicsLinkIt->second.hasSoftBody;
        if (entityHasSoftBody)
        {
            // A renderable currently selects exactly one deformable draw path/metadata family.
            // Supporting both soft-body surface deformation and procedural curve deformation on
            // the same renderable would require splitting them into separate renderables or
            // generalizing the draw metadata model. Until then, skip the curve binding here.
            CRESSIM_LOG_WARNING(
                "Procedural deformable curve rendering is not supported on the same renderable "
                "entity as a soft body. Curve render binding for entity ",
                renderable.entityId, " will be skipped.");
            continue;
        }

        const physics::AuthoredParticleSequenceState *sequence =
            mPhysicsWorld.tryGetParticleSequence(component.sequenceId);
        const graphics::MeshResourceDesc *mesh = resources.tryGetMesh(renderable.mesh);
        if (sequence == nullptr || !sequence->enabled || mesh == nullptr)
        {
            continue;
        }

        const std::uint32_t radialResolution = std::max(component.radialResolution, 3u);
        std::vector<std::uint32_t> resolvedParticleIndices;
        resolvedParticleIndices.reserve(sequence->entries.size());
        bool valid = !sequence->entries.empty();
        for (const physics::AuthoredParticleReference &entry : sequence->entries)
        {
            const std::optional<std::uint32_t> particleIndex = resolveParticleReference(entry);
            if (!particleIndex.has_value())
            {
                valid = false;
                break;
            }
            resolvedParticleIndices.push_back(*particleIndex);
        }

        if (!valid || resolvedParticleIndices.size() < 2u || component.radius <= 0.0f)
        {
            continue;
        }

        const std::uint32_t particleCount =
            static_cast<std::uint32_t>(resolvedParticleIndices.size());
        const std::uint32_t expectedVertexCount = particleCount * radialResolution;
        const std::uint32_t expectedIndexCount  = (particleCount - 1u) * radialResolution * 6u;
        if (mesh->vertices.size() != expectedVertexCount ||
            mesh->indices.size() != expectedIndexCount)
        {
            CRESSIM_LOG_ERROR("Curve render binding build failed for entity ", renderable.entityId,
                              ": visual mesh topology must match the canonical tube layout for ",
                              particleCount, " samples and radial resolution ", radialResolution,
                              ".");
            continue;
        }

        const std::uint32_t curveIndex =
            static_cast<std::uint32_t>(curveRenderData.descriptors.size());
        const std::uint32_t particleIndexStart =
            static_cast<std::uint32_t>(curveRenderData.particleIndices.size());
        const std::uint32_t vertexBase = curveIndex == 0u
                                             ? 0u
                                             : (curveRenderData.descriptors.back().vertexBase +
                                                curveRenderData.descriptors.back().vertexCount);
        curveRenderData.particleIndices.insert(curveRenderData.particleIndices.end(),
                                               resolvedParticleIndices.begin(),
                                               resolvedParticleIndices.end());
        curveRenderData.descriptors.push_back(physics::CurveRenderDescriptorHost{
            particleIndexStart,
            particleCount,
            vertexBase,
            expectedVertexCount,
            radialResolution,
            renderable.envIndex,
            component.radius,
        });

        mCurveRenderVertexBaseByObject[objectIndex]       = vertexBase;
        mCurveRenderVertexNormalBaseByObject[objectIndex] = vertexBase;
        mCurveRenderIndexByObject[objectIndex]            = curveIndex;
        mCurveRenderVertexCountByObject[objectIndex]      = expectedVertexCount;
    }

    for (std::uint32_t objectIndex = 0u;
         objectIndex < static_cast<std::uint32_t>(mRenderables.size()); ++objectIndex)
    {
        if (mRenderables[objectIndex].entityId != common::kInvalidEntityId)
        {
            markRenderableMetadataDirty(objectIndex);
        }
    }

    mCurveRenderBindingsDirty = false;
    mPhysicsWorld.setCurveRenderData(curveRenderData);
}

void World::Impl::refreshDirtyRenderableMetadata(const graphics::RenderResourceManager &resources)
{
    // TODO: some metadata depends on resources too, not just world state:
    // if a mesh or material changes later inside RenderResourceManager, World does
    // not currently know which object slots should become dirty.
    // Currently no supported mutation path exists, but if resources become mutable, World has no
    // dependency tracking to invalidate affected slots.

    bool metadataChanged = false;
    for (const std::uint32_t objectIndex : mDirtyRenderableMetadataIndices)
    {
        if (objectIndex >= mRenderableMetadataHost.size() || objectIndex >= mRenderables.size())
        {
            continue;
        }

        const graphics::RenderableInstance &renderable = mRenderables[objectIndex];
        graphics::GpuRenderableMetadata entry{};
        graphics::GpuRenderableFlags renderableFlags = graphics::GpuRenderableFlags::None;
        entry.deformVertexBase                       = kInvalidSlot;
        entry.deformNormalBase                       = kInvalidSlot;
        entry.deformableIndex                        = kInvalidSlot;
        entry.deformVertexCount                      = 0u;
        entry.deformableType =
            static_cast<std::uint32_t>(graphics::GpuRenderableDeformableType::None);
        entry.segmentationId = renderable.segmentationId;
        entry.entityPoseSlot = mOwner->entityPoseSlot(renderable.entityId);
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
                            entry.deformableIndex = softBodyIndex;
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
                entry.deformVertexBase  = mSoftBodyVertexBindingBaseByObject[objectIndex];
                entry.deformNormalBase  = mSoftBodyVertexNormalBaseByObject[objectIndex];
                entry.deformVertexCount = mSoftBodyVertexCountByObject[objectIndex];
                entry.deformableType =
                    static_cast<std::uint32_t>(graphics::GpuRenderableDeformableType::SoftBody);
            }

            if (const auto curveIt = mProceduralDeformableCurveRenders.find(renderable.entityId);
                curveIt != mProceduralDeformableCurveRenders.end() && curveIt->second.enabled &&
                mCurveRenderIndexByObject[objectIndex] != kInvalidSlot)
            {
                entry.deformVertexBase  = mCurveRenderVertexBaseByObject[objectIndex];
                entry.deformNormalBase  = mCurveRenderVertexNormalBaseByObject[objectIndex];
                entry.deformVertexCount = mCurveRenderVertexCountByObject[objectIndex];
                entry.deformableIndex   = mCurveRenderIndexByObject[objectIndex];
                entry.deformableType =
                    static_cast<std::uint32_t>(graphics::GpuRenderableDeformableType::Curve);
                localBoundsMin = Diligent::float3{0.0f, 0.0f, 0.0f};
                localBoundsMax = Diligent::float3{0.0f, 0.0f, 0.0f};
                hasBounds      = true;
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

        if (std::memcmp(&mRenderableMetadataHost[objectIndex], &entry, sizeof(entry)) != 0)
        {
            mRenderableMetadataHost[objectIndex] = entry;
            metadataChanged                      = true;
        }
    }

    if (metadataChanged)
    {
        bumpGeneration(mRenderableMetadataRevision);
    }
}

void World::Impl::rebuildDrawRegistries(const graphics::RenderResourceManager &resources)
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
                : (mProceduralDeformableCurveRenders.find(renderable.entityId) !=
                               mProceduralDeformableCurveRenders.end() &&
                           mCurveRenderIndexByObject[objectIndex] != kInvalidSlot
                       ? graphics::MaterialProgramFamily::CurveLit
                       : material->pipeline.programFamily);

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

    bumpGeneration(mRenderableQueueInfoRevision);
}

void World::Impl::refreshRenderablePose(std::uint32_t objectIndex)
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
        renderable.worldTransform           = mOwner->tryGetTransform(renderable.entityId)
                                                  .value_or(TransformComponent{})
                                                  .worldTransform;
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
        mOwner->tryGetTransform(renderable.entityId).value_or(TransformComponent{}).worldTransform;
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

void World::Impl::refreshCameraEntry(std::uint32_t cameraIndex)
{
    if (cameraIndex >= mRenderCameras.size())
    {
        return;
    }

    graphics::CameraData &cameraData = mRenderCameras[cameraIndex];
    if (cameraData.entityId == common::kInvalidEntityId || cameraData.cameraSlot == kInvalidSlot)
    {
        const graphics::GpuCameraInput cleared{};
        if (std::memcmp(&mCameraInputsHost[cameraIndex], &cleared, sizeof(cleared)) != 0)
        {
            mCameraInputsHost[cameraIndex] = cleared;
            bumpGeneration(mCameraInputRevision);
            return;
        }
        mCameraInputsHost[cameraIndex] = cleared;
        return;
    }

    cameraData.worldTransform =
        mOwner->tryGetTransform(cameraData.entityId).value_or(TransformComponent{}).worldTransform;

    graphics::GpuCameraInput input{};
    input.position =
        Diligent::float4{cameraData.worldTransform.position.x, cameraData.worldTransform.position.y,
                         cameraData.worldTransform.position.z, 1.0f};
    input.orientation = Diligent::float4{
        cameraData.worldTransform.rotation.q.x, cameraData.worldTransform.rotation.q.y,
        cameraData.worldTransform.rotation.q.z, cameraData.worldTransform.rotation.q.w};
    const std::uint32_t poseSlot = mOwner->entityPoseSlot(cameraData.entityId);
    input.projectionParams = Diligent::float4{cameraData.verticalFovDegrees, cameraData.nearClip,
                                              cameraData.farClip, 0.0f};
    input.viewportAndOutputSize = Diligent::float4{
        cameraData.viewport.width, cameraData.viewport.height,
        static_cast<float>(cameraData.outputWidth), static_cast<float>(cameraData.outputHeight)};
    input.envIndex       = cameraData.envIndex;
    input.cameraSlot     = cameraData.cameraSlot;
    input.active         = 1u;
    input.entityPoseSlot = poseSlot;
    if (std::memcmp(&mCameraInputsHost[cameraIndex], &input, sizeof(input)) != 0)
    {
        mCameraInputsHost[cameraIndex] = input;
        bumpGeneration(mCameraInputRevision);
        return;
    }
    mCameraInputsHost[cameraIndex] = input;
}

void World::Impl::refreshLightEntry(std::uint32_t lightIndex)
{
    if (lightIndex >= mRenderLights.size())
    {
        return;
    }

    graphics::LightData &lightData = mRenderLights[lightIndex];
    if (lightData.entityId == common::kInvalidEntityId || lightData.lightSlot == kInvalidSlot)
    {
        const graphics::GpuLightInput cleared{};
        if (std::memcmp(&mLightInputsHost[lightIndex], &cleared, sizeof(cleared)) != 0)
        {
            mLightInputsHost[lightIndex] = cleared;
            bumpGeneration(mLightInputRevision);
            return;
        }
        mLightInputsHost[lightIndex] = cleared;
        return;
    }

    const TransformComponent transform =
        mOwner->tryGetTransform(lightData.entityId).value_or(TransformComponent{});
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
    input.shadowFadeDistance = lightData.shadowFadeDistance;
    input.shadowBias         = lightData.shadowBias;
    input.envIndex           = lightData.envIndex;
    input.lightSlot          = lightData.lightSlot;
    input.type               = static_cast<std::uint32_t>(lightData.type);
    input.active             = 1u;
    input.castsShadows       = lightData.castsShadows ? 1u : 0u;
    if (std::memcmp(&mLightInputsHost[lightIndex], &input, sizeof(input)) != 0)
    {
        mLightInputsHost[lightIndex] = input;
        bumpGeneration(mLightInputRevision);
        return;
    }
    mLightInputsHost[lightIndex] = input;
}

void World::Impl::rebuildLocalLightSelections()
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

    bumpGeneration(mLocalLightSelectionRevision);
}

bool World::Impl::moveRenderableToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
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

bool World::Impl::moveCameraToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
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

bool World::Impl::moveLightToEnvironment(common::EntityId entityId, std::uint32_t envIndex)
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

bool World::Impl::isLightSlotOccupied(std::uint32_t envIndex, std::uint32_t slot) const noexcept
{
    if (slot >= mSceneLayout.maxLightsPerEnv)
    {
        return false;
    }

    const std::uint32_t lightIndex = envIndex * mSceneLayout.maxLightsPerEnv + slot;
    return lightIndex < mRenderLights.size() && isValidLight(mRenderLights[lightIndex]);
}

std::uint32_t World::Impl::allocateLightSlot(std::uint32_t envIndex,
                                             bool reserveMainDirectionalSlot)
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

bool World::Impl::tryGetLightIndexForType(common::EntityId entityId, graphics::GpuLightType type,
                                          const char *operation,
                                          std::uint32_t &lightIndex) const noexcept
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

bool World::Impl::removeLight(common::EntityId entityId)
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

void World::Impl::clearColliderLinks(common::EntityId entityId) noexcept
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
