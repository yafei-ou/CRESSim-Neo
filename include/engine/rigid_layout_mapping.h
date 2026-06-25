#ifndef CRESSIM_NEO_ENGINE_RIGID_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_RIGID_LAYOUT_MAPPING_H

#include "common/id.h"
#include "engine/export.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

struct CRESSIM_NEO_ENGINE_API RigidLayoutMapping
{
    std::uint32_t rigidBodyCount = 0u;
    std::uint32_t colliderCount  = 0u;
    std::uint64_t layoutRevision = 0u;

    std::vector<physics::RigidBodyId> rigidBodyIds;
    std::vector<common::EntityId> rigidBodyEntityIds;
    std::vector<std::uint32_t> rigidBodyEnvironmentIndices;

    std::vector<physics::ColliderId> colliderIds;
    std::vector<common::EntityId> colliderEntityIds;
    std::vector<physics::RigidBodyId> colliderOwnerBodyIds;
    std::vector<std::uint32_t> colliderOwnerBodyIndices;
    std::vector<std::uint32_t> colliderEnvironmentIndices;
    std::vector<std::uint32_t> colliderShapeTypes;
    std::vector<std::uint32_t> colliderEnabledFlags;
    std::vector<std::uint32_t> colliderCollisionLayers;
    std::vector<std::uint32_t> colliderCollisionMasks;
    std::vector<Diligent::float3> colliderLocalPositions;
    std::vector<Diligent::QuaternionF> colliderLocalRotations;
    std::vector<Diligent::float4> colliderShapeParams;

    std::vector<std::uint32_t> bodyColliderOffsets;
    std::vector<std::uint32_t> bodyColliderCounts;
    std::vector<std::uint32_t> bodyColliderIndices;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RIGID_LAYOUT_MAPPING_H
