#ifndef CRESSIM_NEO_ENGINE_WORLD_H
#define CRESSIM_NEO_ENGINE_WORLD_H

#include "common/id.h"
#include "engine/components.h"
#include "engine/export.h"
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
    void setSceneLayout(const gpu::GpuSceneLayoutDesc &layout);
    const gpu::GpuSceneLayoutDesc &sceneLayout() const noexcept;
    bool setEntityEnvironment(common::EntityId entityId, std::uint32_t envIndex);
    std::uint32_t entityEnvironment(common::EntityId entityId) const noexcept;
    bool setEnvironmentIbl(std::uint32_t envIndex, const graphics::EnvironmentIblDesc &desc);
    const graphics::EnvironmentIblDesc *tryGetEnvironmentIbl(std::uint32_t envIndex) const noexcept;

    bool isAlive(common::EntityId entityId) const;
    const std::vector<common::EntityId> &entities() const noexcept;

    void setTransform(common::EntityId entityId, const TransformComponent &component);
    void setMeshRenderer(common::EntityId entityId, const MeshRendererComponent &component);
    void setCamera(common::EntityId entityId, const CameraComponent &component);
    void setDirectionalLight(common::EntityId entityId, const DirectionalLightComponent &component);

    // Physics is owned by PhysicsWorld now.
    void setRigidBody(common::EntityId entityId, const RigidBodyComponent &component);
    bool removeRigidBody(common::EntityId entityId);

    ColliderHandle addCollider(common::EntityId entityId, const ColliderComponent &component);
    void updateCollider(ColliderHandle handle, const ColliderComponent &component);
    bool removeCollider(ColliderHandle handle);

    bool removeTransform(common::EntityId entityId);
    bool removeMeshRenderer(common::EntityId entityId);
    bool removeCamera(common::EntityId entityId);
    bool removeDirectionalLight(common::EntityId entityId);

    std::optional<TransformComponent> tryGetTransform(common::EntityId entityId) const;
    std::optional<MeshRendererComponent> tryGetMeshRenderer(common::EntityId entityId) const;
    std::optional<CameraComponent> tryGetCamera(common::EntityId entityId) const;
    std::optional<DirectionalLightComponent> tryGetDirectionalLight(
        common::EntityId entityId) const;

    // Read rigid body/collider through physics.
    std::optional<RigidBodyComponent> tryGetRigidBody(common::EntityId entityId) const;
    std::optional<ColliderComponent> tryGetCollider(ColliderHandle handle) const;
    const std::vector<ColliderHandle> &colliderHandles(common::EntityId entityId) const;

    physics::PhysicsWorld &physicsWorld() noexcept;
    const physics::PhysicsWorld &physicsWorld() const noexcept;

    void refreshFromPhysics();
    void setGpuEntityScene(const gpu::GpuEntitySceneView &sceneView) noexcept;

    const std::vector<graphics::RenderableInstance> &renderables() const noexcept;
    const std::vector<graphics::CameraData> &cameras() const noexcept;
    const std::vector<graphics::DirectionalLightData> &directionalLights() const noexcept;
    const std::vector<Diligent::float4> &renderObjectPositions() const noexcept;
    const std::vector<Diligent::float4> &renderObjectOrientations() const noexcept;
    const std::vector<Diligent::float4> &renderObjectScales() const noexcept;
    const std::vector<gpu::GpuRenderableMetadata> &renderableMetadata() const noexcept;
    const std::vector<gpu::GpuRenderableQueueInfo> &renderableQueueInfo() const noexcept;
    const std::vector<gpu::GpuCameraInput> &cameraInputs() const noexcept;
    const std::vector<gpu::GpuDirectionalLightInput> &lightInputs() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &opaqueDrawRegistry() const noexcept;
    const std::vector<graphics::IndirectCommandRegistryEntry> &shadowDrawRegistry() const noexcept;
    const std::vector<gpu::GpuEntityPoseMappingEntry> &physicsRenderableMappings();
    const gpu::GpuEntitySceneView &gpuEntityScene() const noexcept;
    graphics::HostSceneView hostSceneView() const noexcept;
    void ensureRenderStateUpToDate(const graphics::RenderResourceManager &resources);

    // ---------- GPU-friendly SoA views ----------
    struct TransformSoA
    {
        std::vector<common::EntityId> entityIds;
        std::vector<Diligent::float4> positions;
        std::vector<Diligent::float4> rotations;
        std::vector<Diligent::float4> scales;
    };

    const TransformSoA &transformSoA() const noexcept
    {
        return mTransforms;
    }

private:
    static constexpr std::uint32_t kInvalidIndex = 0xffffffffu;

    struct PhysicsLink
    {
        bool hasRigidBody = false;
        std::vector<ColliderHandle> colliders;
    };

    template <typename SoA>
    struct SparseIndex
    {
        std::unordered_map<common::EntityId, std::uint32_t> entityToIndex;
    };

    void ensureEntity(common::EntityId entityId);
    void ensureHostSceneStorage();
    void refreshRenderablePose(std::uint32_t objectIndex);
    void refreshCameraEntry(std::uint32_t cameraIndex);
    void refreshDirectionalLightEntry(std::uint32_t lightIndex);
    void refreshDirtyRenderableMetadata(const graphics::RenderResourceManager &resources);
    void rebuildDrawRegistries(const graphics::RenderResourceManager &resources);
    void clearDirtyIndexSet(std::vector<std::uint32_t> &dirtyIndices,
                            std::vector<std::uint8_t> &dirtyBits);
    void markRenderablePoseDirty(std::uint32_t objectIndex);
    void markCameraDirty(std::uint32_t cameraIndex);
    void markLightDirty(std::uint32_t lightIndex);
    void markRenderableMetadataDirty(std::uint32_t objectIndex);
    void moveRenderableToEnvironment(common::EntityId entityId, std::uint32_t envIndex);
    void moveCameraToEnvironment(common::EntityId entityId, std::uint32_t envIndex);
    void moveDirectionalLightToEnvironment(common::EntityId entityId, std::uint32_t envIndex);

    static Diligent::float4 packPosition(const TransformComponent &c)
    {
        return Diligent::float4{c.worldTransform.position.x, c.worldTransform.position.y,
                                c.worldTransform.position.z, 0.0f};
    }

    static Diligent::float4 packRotation(const TransformComponent &c)
    {
        return Diligent::float4{c.worldTransform.rotation.q.x, c.worldTransform.rotation.q.y,
                                c.worldTransform.rotation.q.z, c.worldTransform.rotation.q.w};
    }

    static Diligent::float4 packScale(const TransformComponent &c)
    {
        return Diligent::float4{c.worldTransform.scale.x, c.worldTransform.scale.y,
                                c.worldTransform.scale.z, 0.0f};
    }

    static TransformComponent unpackTransform(const Diligent::float4 &position,
                                              const Diligent::float4 &rotation,
                                              const Diligent::float4 &scale)
    {
        TransformComponent component{};
        component.worldTransform.position = Diligent::float3{position.x, position.y, position.z};
        component.worldTransform.rotation =
            Diligent::QuaternionF{rotation.x, rotation.y, rotation.z, rotation.w};
        component.worldTransform.scale = Diligent::float3{scale.x, scale.y, scale.z};
        return component;
    }

    template <typename SoAType, typename WriterFn>
    static void upsertSoA(common::EntityId entityId, SoAType &soa, SparseIndex<SoAType> &index,
                          WriterFn &&writer)
    {
        const auto it = index.entityToIndex.find(entityId);
        if (it == index.entityToIndex.end())
        {
            const std::uint32_t newIndex = static_cast<std::uint32_t>(soa.entityIds.size());
            soa.entityIds.push_back(entityId);
            writer(newIndex, true);
            index.entityToIndex.emplace(entityId, newIndex);
            return;
        }

        writer(it->second, false);
    }

    common::EntityId mNextEntityId = 1;
    std::uint32_t mNextColliderId  = 1;

    std::vector<common::EntityId> mEntities;
    std::unordered_set<common::EntityId> mAlive;

    TransformSoA mTransforms{};

    SparseIndex<TransformSoA> mTransformIndex{};

    std::unordered_map<common::EntityId, PhysicsLink> mPhysicsLinks{};
    std::unordered_map<std::uint32_t, common::EntityId> mColliderOwnerEntity{};

    physics::PhysicsWorld mPhysicsWorld{};
    gpu::GpuSceneLayoutDesc mSceneLayout{};
    std::unordered_map<common::EntityId, std::uint32_t> mEntityEnvironments{};

    std::vector<graphics::RenderableInstance> mRenderables{};
    std::vector<graphics::CameraData> mRenderCameras{};
    std::vector<graphics::DirectionalLightData> mRenderDirectionalLights{};
    std::vector<Diligent::float4> mRenderObjectPositions{};
    std::vector<Diligent::float4> mRenderObjectOrientations{};
    std::vector<Diligent::float4> mRenderObjectScales{};
    std::vector<gpu::GpuRenderableMetadata> mRenderableMetadataHost{};
    std::vector<gpu::GpuRenderableQueueInfo> mRenderableQueueInfoHost{};
    std::vector<gpu::GpuCameraInput> mCameraInputsHost{};
    std::vector<gpu::GpuDirectionalLightInput> mLightInputsHost{};
    std::vector<graphics::EnvironmentIblDesc> mEnvironmentIbls{};
    std::vector<graphics::IndirectCommandRegistryEntry> mOpaqueDrawRegistryHost{};
    std::vector<graphics::IndirectCommandRegistryEntry> mShadowDrawRegistryHost{};
    gpu::GpuEntitySceneView mGpuEntityScene{};
    std::vector<gpu::GpuEntityPoseMappingEntry> mPhysicsRenderableMappingsCache{};

    std::unordered_map<common::EntityId, std::size_t> mRenderableIndices{};
    std::unordered_map<common::EntityId, std::size_t> mRenderCameraIndices{};
    std::unordered_map<common::EntityId, std::size_t> mRenderDirectionalLightIndices{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextRenderableSlotByEnv{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextCameraSlotByEnv{};
    std::unordered_map<std::uint32_t, std::uint32_t> mNextDirectionalLightSlotByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeRenderableSlotsByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeCameraSlotsByEnv{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> mFreeDirectionalLightSlotsByEnv{};
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

    std::uint64_t mCachedPhysicsRenderableMappingsBodyTopologyRevision = ~0ull;
};

} // namespace cressim::neo::engine

#endif
