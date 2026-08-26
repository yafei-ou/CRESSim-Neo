#ifndef CRESSIM_NEO_ENGINE_RIGID_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_RIGID_LAYOUT_MAPPING_H

#include "common/id.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

/// @brief Prepared parallel arrays mapping rigid bodies and colliders to their GPU slot layout.
struct RigidLayoutMapping
{
    std::uint32_t rigidBodyCount = 0u; ///< Number of rigid-body slots.
    std::uint32_t colliderCount  = 0u; ///< Number of collider slots.
    /// Prepared host-side rigid/collider slot-layout invalidation key produced by prepare().
    /// This describes when authored slot interpretation changes and is not the same as the live
    /// GPU custom-compute resource bindingGeneration exposed after uploadWorld().
    std::uint64_t layoutRevision = 0u;

    /// @brief Rigid-body IDs, indexed by rigid-body slot in [0, rigidBodyCount).
    std::vector<physics::RigidBodyId> rigidBodyIds;
    std::vector<common::EntityId> rigidBodyEntityIds;       ///< Entities owning rigid bodies.
    std::vector<std::uint32_t> rigidBodyEnvironmentIndices; ///< Environment indices.

    /// @brief Collider IDs, indexed by collider slot in [0, colliderCount).
    std::vector<physics::ColliderId> colliderIds;
    std::vector<common::EntityId> colliderEntityIds;        ///< Entities owning colliders.
    std::vector<physics::RigidBodyId> colliderOwnerBodyIds; ///< Owning rigid-body IDs.
    std::vector<std::uint32_t> colliderOwnerBodyIndices;    ///< Owning rigid-body slots.
    std::vector<std::uint32_t> colliderEnvironmentIndices;  ///< Environment indices.
    std::vector<std::uint32_t> colliderShapeTypes;          ///< Numeric ColliderShapeType values.
    std::vector<std::uint32_t> colliderEnabledFlags; ///< One for enabled colliders, zero otherwise.
    std::vector<std::uint32_t> colliderCollisionLayers;        ///< Collision layer masks.
    std::vector<std::uint32_t> colliderCollisionMasks;         ///< Collision filter masks.
    std::vector<Diligent::float3> colliderLocalPositions;      ///< Body-local positions.
    std::vector<Diligent::QuaternionF> colliderLocalRotations; ///< Body-local orientations.
    std::vector<Diligent::float4> colliderShapeParams;         ///< Shape-specific parameters.

    /// @brief First flattened collider index per body, indexed by rigid-body slot.
    std::vector<std::uint32_t> bodyColliderOffsets;
    std::vector<std::uint32_t> bodyColliderCounts;  ///< Collider count per body.
    std::vector<std::uint32_t> bodyColliderIndices; ///< Flattened collider slots.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RIGID_LAYOUT_MAPPING_H
