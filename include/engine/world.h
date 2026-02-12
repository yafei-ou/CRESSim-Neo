#ifndef CRESSIM_NEO_ENGINE_WORLD_H
#define CRESSIM_NEO_ENGINE_WORLD_H

#include "common/id.h"
#include "engine/components.h"
#include "engine/export.h"

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

    TransformComponent& setTransform(common::EntityId entityId, const TransformComponent& component);
    MeshRendererComponent& setMeshRenderer(common::EntityId entityId, const MeshRendererComponent& component);
    CameraComponent& setCamera(common::EntityId entityId, const CameraComponent& component);
    DirectionalLightComponent& setDirectionalLight(common::EntityId entityId, const DirectionalLightComponent& component);

    const TransformComponent* tryGetTransform(common::EntityId entityId) const;
    const MeshRendererComponent* tryGetMeshRenderer(common::EntityId entityId) const;
    const CameraComponent* tryGetCamera(common::EntityId entityId) const;
    const DirectionalLightComponent* tryGetDirectionalLight(common::EntityId entityId) const;

private:
    void ensureEntity(common::EntityId entityId);

    common::EntityId mNextEntityId = 1;
    std::vector<common::EntityId> mEntities;
    std::unordered_set<common::EntityId> mAlive;

    std::unordered_map<common::EntityId, TransformComponent> mTransforms;
    std::unordered_map<common::EntityId, MeshRendererComponent> mMeshRenderers;
    std::unordered_map<common::EntityId, CameraComponent> mCameras;
    std::unordered_map<common::EntityId, DirectionalLightComponent> mDirectionalLights;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_WORLD_H
