#ifndef CRESSIM_NEO_ENGINE_WORLD_H
#define CRESSIM_NEO_ENGINE_WORLD_H

#include "common/id.h"
#include "engine/components.h"
#include "engine/export.h"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cressim::neo::engine
{

class CRESSIM_NEO_ENGINE_API World
{
public:
    common::EntityId createEntity();
    bool destroyEntity(common::EntityId entityId);

    bool isAlive(common::EntityId entityId) const;
    const std::vector<common::EntityId>& entities() const noexcept;

    TransformComponent& setTransform(common::EntityId entityId,
                                     const TransformComponent& component);
    MeshRendererComponent& setMeshRenderer(common::EntityId entityId,
                                           const MeshRendererComponent& component);
    CameraComponent& setCamera(common::EntityId entityId, const CameraComponent& component);
    DirectionalLightComponent& setDirectionalLight(common::EntityId entityId,
                                                   const DirectionalLightComponent& component);
    RigidBodyComponent& setRigidBody(common::EntityId entityId,
                                     const RigidBodyComponent& component);
    ColliderHandle addCollider(common::EntityId entityId, const ColliderComponent& component);
    ColliderComponent& updateCollider(ColliderHandle handle, const ColliderComponent& component);
    bool removeTransform(common::EntityId entityId);
    bool removeMeshRenderer(common::EntityId entityId);
    bool removeCamera(common::EntityId entityId);
    bool removeDirectionalLight(common::EntityId entityId);
    bool removeRigidBody(common::EntityId entityId);
    bool removeCollider(ColliderHandle handle);

    const TransformComponent* tryGetTransform(common::EntityId entityId) const;
    const MeshRendererComponent* tryGetMeshRenderer(common::EntityId entityId) const;
    const CameraComponent* tryGetCamera(common::EntityId entityId) const;
    const DirectionalLightComponent* tryGetDirectionalLight(common::EntityId entityId) const;
    const RigidBodyComponent* tryGetRigidBody(common::EntityId entityId) const;
    const ColliderComponent* tryGetCollider(ColliderHandle handle) const;
    const std::vector<ColliderHandle>& colliderHandles(common::EntityId entityId) const;

    std::uint64_t revision() const noexcept;
    const std::vector<common::EntityId>& dirtyEntities() const noexcept;
    void clearDirtyEntities() noexcept;

private:
    struct ColliderRecord
    {
        common::EntityId ownerEntityId = common::kInvalidEntityId;
        ColliderComponent component{};
    };

    void removeCollidersForEntity(common::EntityId entityId);
    void ensureEntity(common::EntityId entityId);
    void markDirty(common::EntityId entityId);

    common::EntityId mNextEntityId = 1;
    std::uint32_t mNextColliderId  = 1;
    std::vector<common::EntityId> mEntities;
    std::unordered_set<common::EntityId> mAlive;

    std::unordered_map<common::EntityId, TransformComponent> mTransforms;
    std::unordered_map<common::EntityId, MeshRendererComponent> mMeshRenderers;
    std::unordered_map<common::EntityId, CameraComponent> mCameras;
    std::unordered_map<common::EntityId, DirectionalLightComponent> mDirectionalLights;
    std::unordered_map<common::EntityId, RigidBodyComponent> mRigidBodies;
    std::unordered_map<common::EntityId, std::vector<ColliderHandle>> mEntityColliderHandles;
    std::unordered_map<std::uint32_t, ColliderRecord> mColliders;
    std::uint64_t mRevision = 0;
    std::vector<common::EntityId> mDirtyEntities;
    std::unordered_set<common::EntityId> mDirtySet;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_WORLD_H
