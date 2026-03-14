#ifndef CRESSIM_NEO_ENGINE_WORLD_H
#define CRESSIM_NEO_ENGINE_WORLD_H

#include "common/id.h"
#include "engine/components.h"
#include "engine/export.h"
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

    common::EntityId createEntity();
    bool destroyEntity(common::EntityId entityId);

    bool isAlive(common::EntityId entityId) const;
    const std::vector<common::EntityId>& entities() const noexcept;

    void setTransform(common::EntityId entityId, const TransformComponent& component);
    void setMeshRenderer(common::EntityId entityId, const MeshRendererComponent& component);
    void setCamera(common::EntityId entityId, const CameraComponent& component);
    void setDirectionalLight(common::EntityId entityId,
                             const DirectionalLightComponent& component);

    // Physics is owned by PhysicsWorld now.
    void setRigidBody(common::EntityId entityId, const RigidBodyComponent& component);
    bool removeRigidBody(common::EntityId entityId);

    ColliderHandle addCollider(common::EntityId entityId, const ColliderComponent& component);
    void updateCollider(ColliderHandle handle, const ColliderComponent& component);
    bool removeCollider(ColliderHandle handle);

    bool removeTransform(common::EntityId entityId);
    bool removeMeshRenderer(common::EntityId entityId);
    bool removeCamera(common::EntityId entityId);
    bool removeDirectionalLight(common::EntityId entityId);

    std::optional<TransformComponent> tryGetTransform(common::EntityId entityId) const;
    std::optional<MeshRendererComponent> tryGetMeshRenderer(common::EntityId entityId) const;
    std::optional<CameraComponent> tryGetCamera(common::EntityId entityId) const;
    std::optional<DirectionalLightComponent> tryGetDirectionalLight(common::EntityId entityId) const;

    // Read rigid body/collider through physics.
    std::optional<RigidBodyComponent> tryGetRigidBody(common::EntityId entityId) const;
    std::optional<ColliderComponent> tryGetCollider(ColliderHandle handle) const;
    const std::vector<ColliderHandle>& colliderHandles(common::EntityId entityId) const;

    physics::PhysicsWorld& physicsWorld() noexcept;
    const physics::PhysicsWorld& physicsWorld() const noexcept;

    void refreshFromPhysics();

    std::uint64_t renderRevision() const noexcept;
    const std::vector<common::EntityId>& renderDirtyEntities() const noexcept;
    void clearRenderDirtyEntities() noexcept;

    // ---------- GPU-friendly SoA views ----------
    struct TransformSoA
    {
        std::vector<common::EntityId> entityIds;
        std::vector<Diligent::float4> positions;
        std::vector<Diligent::float4> rotations;
        std::vector<Diligent::float4> scales;
    };

    struct CameraSoA
    {
        std::vector<common::EntityId> entityIds;
        std::vector<Diligent::float4> projection0; // x=fovYRadians, y=aspect, z=near, w=far
        std::vector<Diligent::float4> projection1; // reserved for jitter / flags / future
        std::vector<std::uint32_t> outputTargetIds;
        std::vector<std::uint32_t> outputWidths;
        std::vector<std::uint32_t> outputHeights;
        std::vector<Diligent::float4> viewports; // x, y, width, height
        std::vector<std::uint32_t> renderOrders;
    };

    struct DirectionalLightSoA
    {
        std::vector<common::EntityId> entityIds;
        std::vector<Diligent::float4> directionsIntensities; // xyz=dir, w=intensity
        std::vector<Diligent::float4> colors;                // xyz=color, w=unused
        std::vector<Diligent::float4> shadowParams; // x=distance, y=fadeDistance
    };

    struct MeshRendererSoA
    {
        std::vector<common::EntityId> entityIds;
        std::vector<std::uint32_t> meshIds;
        std::vector<std::uint32_t> materialIds;
        std::vector<std::uint32_t> visibleFlags;
    };

    const TransformSoA& transformSoA() const noexcept
    {
        return mTransforms;
    }
    const CameraSoA& cameraSoA() const noexcept
    {
        return mCameras;
    }
    const DirectionalLightSoA& directionalLightSoA() const noexcept
    {
        return mDirectionalLights;
    }
    const MeshRendererSoA& meshRendererSoA() const noexcept
    {
        return mMeshRenderers;
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
    void markRenderDirty(common::EntityId entityId);

    static Diligent::float4 packPosition(const TransformComponent& c)
    {
        return Diligent::float4{c.worldTransform.position.x, c.worldTransform.position.y,
                                c.worldTransform.position.z, 0.0f};
    }

    static Diligent::float4 packRotation(const TransformComponent& c)
    {
        return Diligent::float4{c.worldTransform.rotation.q.x, c.worldTransform.rotation.q.y,
                                c.worldTransform.rotation.q.z, c.worldTransform.rotation.q.w};
    }

    static Diligent::float4 packScale(const TransformComponent& c)
    {
        return Diligent::float4{c.worldTransform.scale.x, c.worldTransform.scale.y,
                                c.worldTransform.scale.z, 0.0f};
    }

    static TransformComponent unpackTransform(const Diligent::float4& position,
                                              const Diligent::float4& rotation,
                                              const Diligent::float4& scale)
    {
        TransformComponent component{};
        component.worldTransform.position = Diligent::float3{position.x, position.y, position.z};
        component.worldTransform.rotation =
            Diligent::QuaternionF{rotation.x, rotation.y, rotation.z, rotation.w};
        component.worldTransform.scale = Diligent::float3{scale.x, scale.y, scale.z};
        return component;
    }

    static Diligent::float4 packCameraProjection0(const CameraComponent& c)
    {
        return Diligent::float4{c.verticalFovDegrees, c.aspectRatio, c.nearClip, c.farClip};
    }

    static Diligent::float4 packCameraProjection1(const CameraComponent&)
    {
        return Diligent::float4{0, 0, 0, 0};
    }

    static CameraComponent unpackCamera(const Diligent::float4& projection0,
                                        std::uint32_t outputTargetId, std::uint32_t outputWidth,
                                        std::uint32_t outputHeight,
                                        const Diligent::float4& viewport,
                                        std::uint32_t renderOrder)
    {
        CameraComponent component{};
        component.verticalFovDegrees = projection0.x;
        component.aspectRatio        = projection0.y;
        component.nearClip           = projection0.z;
        component.farClip            = projection0.w;
        component.outputTarget.id    = outputTargetId;
        component.outputWidth        = outputWidth;
        component.outputHeight       = outputHeight;
        component.viewport =
            gpu::GpuRenderViewport{viewport.x, viewport.y, viewport.z, viewport.w};
        component.renderOrder = renderOrder;
        return component;
    }

    static Diligent::float4 packLightDirectionIntensity(const DirectionalLightComponent& c)
    {
        return Diligent::float4{c.direction.x, c.direction.y, c.direction.z, c.intensity};
    }

    static Diligent::float4 packLightColor(const DirectionalLightComponent& c)
    {
        return Diligent::float4{c.color.x, c.color.y, c.color.z, 0.0f};
    }

    static Diligent::float4 packLightShadowParams(const DirectionalLightComponent& c)
    {
        return Diligent::float4{c.shadowDistance, c.shadowFadeDistance, 0.0f, 0.0f};
    }

    static DirectionalLightComponent unpackDirectionalLight(const Diligent::float4& direction,
                                                            const Diligent::float4& color,
                                                            const Diligent::float4& shadowParams)
    {
        DirectionalLightComponent component{};
        component.direction = Diligent::float3{direction.x, direction.y, direction.z};
        component.intensity = direction.w;
        component.color     = Diligent::float3{color.x, color.y, color.z};
        component.shadowDistance     = shadowParams.x;
        component.shadowFadeDistance = shadowParams.y;
        return component;
    }

    template <typename SoAType, typename WriterFn>
    static void upsertSoA(common::EntityId entityId, SoAType& soa, SparseIndex<SoAType>& index,
                          WriterFn&& writer)
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

    template <typename SoAType>
    static bool removeFromSoA(common::EntityId entityId, SoAType& soa, SparseIndex<SoAType>& index);

    common::EntityId mNextEntityId = 1;
    std::uint32_t mNextColliderId  = 1;

    std::vector<common::EntityId> mEntities;
    std::unordered_set<common::EntityId> mAlive;

    TransformSoA mTransforms{};
    CameraSoA mCameras{};
    DirectionalLightSoA mDirectionalLights{};
    MeshRendererSoA mMeshRenderers{};

    SparseIndex<TransformSoA> mTransformIndex{};
    SparseIndex<CameraSoA> mCameraIndex{};
    SparseIndex<DirectionalLightSoA> mDirectionalLightIndex{};
    SparseIndex<MeshRendererSoA> mMeshRendererIndex{};

    std::unordered_map<common::EntityId, PhysicsLink> mPhysicsLinks{};
    std::unordered_map<std::uint32_t, common::EntityId> mColliderOwnerEntity{};

    physics::PhysicsWorld mPhysicsWorld{};

    std::uint64_t mRenderRevision = 0;
    std::vector<common::EntityId> mRenderDirtyEntities;
    std::unordered_set<common::EntityId> mRenderDirtySet;

};

} // namespace cressim::neo::engine

#endif
