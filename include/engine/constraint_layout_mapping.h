#ifndef CRESSIM_NEO_ENGINE_CONSTRAINT_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_CONSTRAINT_LAYOUT_MAPPING_H

#include "common/id.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

/// @brief Prepared parallel arrays describing rigid-particle attachment constraints.
///
/// Every vector except where noted is indexed by constraint slot in [0, count).
struct RigidParticleAttachmentConstraintLayoutMapping
{
    std::uint32_t count = 0u; ///< Number of attachment-constraint slots.

    /// This prepared mapping reflects authored host-side references resolved by prepare().
    /// Runtime GPU edits to the live rigid-particle attachment descriptor buffer after
    /// uploadWorld() (for example retargeting `particleIndex`) do not feed back into these
    /// arrays until the host reauthors and prepares again.
    std::vector<physics::RigidParticleAttachmentConstraintId> constraintIds;
    std::vector<std::uint32_t> environmentIndices;     ///< Owning rigid body's environment indices.
    std::vector<physics::RigidBodyId> rigidBodyIds;    ///< Referenced rigid-body IDs.
    std::vector<std::uint32_t> rigidBodyIndices;       ///< Prepared rigid-body slot indices.
    std::vector<common::EntityId> particleEntityIds;   ///< Entities owning referenced particles.
    std::vector<std::uint32_t> particleReferenceTypes; ///< Numeric particle-reference types.
    std::vector<std::uint32_t> particleLocalIndices;   ///< Particle indices local to their owners.
    std::vector<std::uint32_t> enabledFlags; ///< One for enabled constraints and zero otherwise.
};

/// @brief Prepared parallel arrays describing rigid-distance constraints.
///
/// Every vector is indexed by constraint slot in [0, count).
struct RigidDistanceConstraintLayoutMapping
{
    std::uint32_t count = 0u; ///< Number of rigid-distance constraint slots.

    std::vector<physics::RigidDistanceConstraintId> constraintIds; ///< Constraint IDs.
    std::vector<std::uint32_t> environmentIndices;                 ///< Environment of body A.
    std::vector<physics::RigidBodyId> rigidBodyIdsA;               ///< First rigid-body IDs.
    std::vector<physics::RigidBodyId> rigidBodyIdsB;               ///< Second rigid-body IDs.
    std::vector<std::uint32_t> rigidBodyIndicesA;                  ///< Prepared slots for body A.
    std::vector<std::uint32_t> rigidBodyIndicesB;                  ///< Prepared slots for body B.
    std::vector<std::uint32_t> enabledFlags; ///< One for enabled constraints and zero otherwise.
};

/// @brief Prepared arrays describing routed-cable constraints and their flattened route points.
struct RoutedCableConstraintLayoutMapping
{
    std::uint32_t count = 0u; ///< Number of routed-cable constraint slots.

    /// @brief Constraint IDs, indexed by constraint slot in [0, count).
    std::vector<physics::RoutedCableConstraintId> constraintIds;
    std::vector<std::uint32_t> environmentIndices; ///< First resolved route-point environment.
    std::vector<std::uint32_t> routePointOffsets;  ///< Offsets into flattened route-point arrays.
    std::vector<std::uint32_t> routePointCounts;   ///< Number of route points per constraint.
    std::vector<std::uint32_t> enabledFlags; ///< One for enabled constraints and zero otherwise.
    /// @brief Route-point body IDs in the flattened arrays addressed by offsets and counts above.
    std::vector<physics::RigidBodyId> routePointRigidBodyIds;
    std::vector<std::uint32_t> routePointRigidBodyIndices;     ///< Prepared route-point body slots.
    std::vector<Diligent::float3> routePointLocalGuideOffsets; ///< Body-local guide offsets.
};

/// @brief Complete prepared mapping for the supported rigid-adjacent constraint families.
struct ConstraintLayoutMapping
{
    /// Prepared host-side constraint-layout invalidation key produced by prepare().
    /// This describes when authored slot interpretation changes and is not the same as the live
    /// GPU custom-compute resource bindingGeneration exposed after uploadWorld().
    std::uint64_t layoutRevision = 0u;

    RigidParticleAttachmentConstraintLayoutMapping rigidParticleAttachments; ///< Attachments.
    RigidDistanceConstraintLayoutMapping rigidDistanceConstraints;           ///< Distances.
    RoutedCableConstraintLayoutMapping routedCables;                         ///< Routed cables.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_CONSTRAINT_LAYOUT_MAPPING_H
