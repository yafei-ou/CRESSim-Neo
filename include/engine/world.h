#ifndef CRESSIM_NEO_ENGINE_WORLD_H
#define CRESSIM_NEO_ENGINE_WORLD_H

#include "common/id.h"
#include "engine/components.h"
#include "engine/export.h"
#include "engine/render_scene_types.h"
#include "graphics/host_scene.h"
#include "physics/physics_world.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cressim::neo::engine
{

class CRESSIM_NEO_ENGINE_API World
{
public:
    using ColliderHandle = engine::ColliderHandle;

    common::EntityId createEntity(std::uint32_t envIndex = 0u);
    bool destroyEntity(common::EntityId entityId);
    void setSceneLayout(const common::SceneLayoutDesc &layout);
    const common::SceneLayoutDesc &sceneLayout() const noexcept;
    bool setEntityEnvironment(common::EntityId entityId, std::uint32_t envIndex);
    std::uint32_t entityEnvironment(common::EntityId entityId) const noexcept;
    bool setEnvironmentIbl(std::uint32_t envIndex, const graphics::EnvironmentIblDesc &desc);
    const graphics::EnvironmentIblDesc *tryGetEnvironmentIbl(std::uint32_t envIndex) const noexcept;
    bool setEnvironmentFluid(std::uint32_t envIndex, const graphics::EnvironmentFluidDesc &desc);
    const graphics::EnvironmentFluidDesc *tryGetEnvironmentFluid(
        std::uint32_t envIndex) const noexcept;

    bool isAlive(common::EntityId entityId) const;
    const std::vector<common::EntityId> &entities() const noexcept;

    void setTransform(common::EntityId entityId, const TransformComponent &component);
    void setMeshRenderer(common::EntityId entityId, const MeshRendererComponent &component);
    void setCamera(common::EntityId entityId, const CameraComponent &component);
    void setDirectionalLight(common::EntityId entityId, const DirectionalLightComponent &component);
    void setPointLight(common::EntityId entityId, const PointLightComponent &component);
    void setSpotLight(common::EntityId entityId, const SpotLightComponent &component);

    // Physics is owned by PhysicsWorld now.
    void setRigidBody(common::EntityId entityId, const RigidBodyComponent &component);
    bool removeRigidBody(common::EntityId entityId);
    bool setSoftBody(common::EntityId entityId, const SoftBodyComponent &component);
    bool removeSoftBody(common::EntityId entityId);
    bool setStrand(common::EntityId entityId, const StrandComponent &component);
    bool removeStrand(common::EntityId entityId);
    bool setFluid(common::EntityId entityId, const FluidComponent &component);
    bool removeFluid(common::EntityId entityId);
    physics::AuthoredParticleDistanceConstraintState &upsertParticleDistanceConstraint(
        const physics::AuthoredParticleDistanceConstraintState &state);
    physics::AuthoredSuturingSequenceState &upsertSuturingSequence(
        const physics::AuthoredSuturingSequenceState &state);
    bool removeParticleDistanceConstraint(physics::ParticleConstraintId constraintId);
    bool removeSuturingSequence(physics::SuturingSequenceId sequenceId);
    void setUltrasoundProbe(common::EntityId entityId, const UltrasoundProbeComponent &component);
    bool removeUltrasoundProbe(common::EntityId entityId);
    void setUltrasoundScattererSource(common::EntityId entityId,
                                      const UltrasoundScattererSourceComponent &component);
    bool removeUltrasoundScattererSource(common::EntityId entityId);
    void setUltrasoundScattererAmplitudeRanges(common::EntityId entityId,
                                               const std::vector<UltrasoundAmplitudeRange> &ranges);
    bool clearUltrasoundScattererAmplitudeRanges(common::EntityId entityId);

    ColliderHandle addCollider(common::EntityId entityId, const ColliderComponent &component);
    void updateCollider(ColliderHandle handle, const ColliderComponent &component);
    bool removeCollider(ColliderHandle handle);

    bool removeTransform(common::EntityId entityId);
    bool removeMeshRenderer(common::EntityId entityId);
    bool removeCamera(common::EntityId entityId);
    bool removeDirectionalLight(common::EntityId entityId);
    bool removePointLight(common::EntityId entityId);
    bool removeSpotLight(common::EntityId entityId);

    std::optional<TransformComponent> tryGetTransform(common::EntityId entityId) const;
    std::optional<MeshRendererComponent> tryGetMeshRenderer(common::EntityId entityId) const;
    std::optional<CameraComponent> tryGetCamera(common::EntityId entityId) const;
    std::optional<DirectionalLightComponent> tryGetDirectionalLight(
        common::EntityId entityId) const;
    std::optional<PointLightComponent> tryGetPointLight(common::EntityId entityId) const;
    std::optional<SpotLightComponent> tryGetSpotLight(common::EntityId entityId) const;

    // Read rigid body/collider through physics.
    std::optional<RigidBodyComponent> tryGetRigidBody(common::EntityId entityId) const;
    std::optional<SoftBodyComponent> tryGetSoftBody(common::EntityId entityId) const;
    std::optional<StrandComponent> tryGetStrand(common::EntityId entityId) const;
    std::optional<FluidComponent> tryGetFluid(common::EntityId entityId) const;
    const physics::AuthoredParticleDistanceConstraintState *tryGetParticleDistanceConstraint(
        physics::ParticleConstraintId constraintId) const noexcept;
    const physics::AuthoredSuturingSequenceState *tryGetSuturingSequence(
        physics::SuturingSequenceId sequenceId) const noexcept;
    std::optional<UltrasoundProbeComponent> tryGetUltrasoundProbe(common::EntityId entityId) const;
    std::optional<UltrasoundScattererSourceComponent> tryGetUltrasoundScattererSource(
        common::EntityId entityId) const;
    std::optional<SoftBodyAuthoringParticles> tryGetSoftBodyAuthoringParticles(
        common::EntityId entityId) const;
    const std::vector<UltrasoundAmplitudeRange> *tryGetUltrasoundScattererAmplitudeRanges(
        common::EntityId entityId) const noexcept;
    const UltrasoundProbeResult *tryGetUltrasoundProbeResult(
        common::EntityId entityId) const noexcept;
    std::optional<ColliderComponent> tryGetCollider(ColliderHandle handle) const;
    const std::vector<ColliderHandle> &colliderHandles(common::EntityId entityId) const;

    physics::PhysicsWorld &physicsWorld() noexcept;
    const physics::PhysicsWorld &physicsWorld() const noexcept;

    void setGpuEntityScene(const graphics::GpuEntitySceneView &sceneView) noexcept;

    const std::vector<graphics::RenderableInstance> &renderables() const noexcept;
    const std::vector<graphics::CameraData> &cameras() const noexcept;
    const std::vector<graphics::LightData> &lights() const noexcept;
    const std::vector<Diligent::float4> &renderObjectPositions() const noexcept;
    const std::vector<Diligent::float4> &renderObjectOrientations() const noexcept;
    const std::vector<Diligent::float4> &renderObjectScales() const noexcept;
    const std::vector<graphics::GpuRenderableMetadata> &renderableMetadata() const noexcept;
    const std::vector<graphics::GpuRenderableQueueInfo> &renderableQueueInfo() const noexcept;
    const std::vector<graphics::GpuCameraInput> &cameraInputs() const noexcept;
    const std::vector<graphics::GpuLightInput> &lightInputs() const noexcept;
    const std::vector<graphics::GpuLocalLightSelection> &localLightSelections() const noexcept;
    const std::vector<graphics::GpuSoftBodyVertexBinding> &softBodyVertexBindings() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &opaqueDrawRegistry() const noexcept;
    const std::vector<graphics::TransparentDrawEntry> &transparentDrawRegistry() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &shadowDrawRegistry() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &localShadowDrawRegistry()
        const noexcept;
    const std::vector<EntityPoseMappingEntry> &physicsRenderableMappings();
    const graphics::GpuEntitySceneView &gpuEntityScene() const noexcept;
    graphics::HostSceneView hostSceneView() const noexcept;
    void ensureRenderStateUpToDate(const graphics::RenderResourceManager &resources);
    const std::unordered_map<common::EntityId, UltrasoundProbeComponent> &
    ultrasoundProbeComponents() const noexcept;
    const std::unordered_map<common::EntityId, UltrasoundScattererSourceComponent> &
    ultrasoundScattererSourceComponents() const noexcept;
    std::uint64_t ultrasoundScattererAmplitudeRevision() const noexcept;
    void setUltrasoundProbeResult(common::EntityId entityId, const UltrasoundProbeResult &result);
    void clearUltrasoundProbeResult(common::EntityId entityId);

private:
    static constexpr std::uint32_t kInvalidIndex = 0xffffffffu;

    struct PhysicsLink
    {
        bool hasRigidBody = false;
        bool hasSoftBody  = false;
        bool hasStrand    = false;
        bool hasFluid     = false;
        std::vector<ColliderHandle> colliders;
    };

    [[nodiscard]] bool requireAliveEntity(common::EntityId entityId,
                                          const char *operation) const noexcept;
    void ensureHostSceneStorage();
    void refreshRenderablePose(std::uint32_t objectIndex);
    void refreshCameraEntry(std::uint32_t cameraIndex);
    void refreshLightEntry(std::uint32_t lightIndex);
    void rebuildLocalLightSelections();
    void rebuildSoftBodyRenderBindings(const graphics::RenderResourceManager &resources);
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

    common::EntityId mNextEntityId = 1;
    std::uint32_t mNextColliderId  = 1;

    std::vector<common::EntityId> mEntities;
    std::unordered_set<common::EntityId> mAlive;

    struct TransformStorage
    {
        std::vector<common::EntityId> entityIds;
        std::vector<TransformComponent> components;
    };

    TransformStorage mTransforms{};
    std::unordered_map<common::EntityId, std::uint32_t> mTransformIndex{};

    std::unordered_map<common::EntityId, PhysicsLink> mPhysicsLinks{};
    std::unordered_map<std::uint32_t, common::EntityId> mColliderOwnerEntity{};
    std::unordered_map<common::EntityId, UltrasoundProbeComponent> mUltrasoundProbes{};
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
    // All light types share one unified light slot space per environment. Each entity may own at
    // most one light entry in this storage. Slot 0 remains reserved for the main directional
    // light when requested.
    std::unordered_map<common::EntityId, std::size_t> mRenderLightIndices{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextRenderableSlotByEnv{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextCameraSlotByEnv{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextLightSlotByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeRenderableSlotsByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeCameraSlotsByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeLightSlotsByEnv{};
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
    std::vector<std::uint32_t> mSoftBodyVertexBindingBaseByObject{};
    std::vector<std::uint32_t> mSoftBodyVertexNormalBaseByObject{};
    std::vector<std::uint32_t> mSoftBodyVertexCountByObject{};

    std::uint64_t mCachedPhysicsRenderableMappingsBodyTopologyRevision = ~0ull;
    std::uint64_t mCachedSoftBodyRenderTopologyRevision                = ~0ull;
    std::uint64_t mCachedSoftBodyPhysicsRevision                       = ~0ull;
};

} // namespace cressim::neo::engine

#endif
