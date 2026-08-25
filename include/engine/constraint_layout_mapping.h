#ifndef CRESSIM_NEO_ENGINE_CONSTRAINT_LAYOUT_MAPPING_H
#define CRESSIM_NEO_ENGINE_CONSTRAINT_LAYOUT_MAPPING_H

#include "common/id.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

struct RigidParticleAttachmentConstraintLayoutMapping
{
    std::uint32_t count = 0u;

    /// This prepared mapping reflects authored host-side references resolved by prepare().
    /// Runtime GPU edits to the live rigid-particle attachment descriptor buffer after
    /// uploadWorld() (for example retargeting `particleIndex`) do not feed back into these
    /// arrays until the host reauthors and prepares again.
    std::vector<physics::RigidParticleAttachmentConstraintId> constraintIds;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<physics::RigidBodyId> rigidBodyIds;
    std::vector<std::uint32_t> rigidBodyIndices;
    std::vector<common::EntityId> particleEntityIds;
    std::vector<std::uint32_t> particleReferenceTypes;
    std::vector<std::uint32_t> particleLocalIndices;
    std::vector<std::uint32_t> enabledFlags;
};

struct RigidDistanceConstraintLayoutMapping
{
    std::uint32_t count = 0u;

    std::vector<physics::RigidDistanceConstraintId> constraintIds;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<physics::RigidBodyId> rigidBodyIdsA;
    std::vector<physics::RigidBodyId> rigidBodyIdsB;
    std::vector<std::uint32_t> rigidBodyIndicesA;
    std::vector<std::uint32_t> rigidBodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
};

struct RoutedCableConstraintLayoutMapping
{
    std::uint32_t count = 0u;

    std::vector<physics::RoutedCableConstraintId> constraintIds;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<std::uint32_t> routePointOffsets;
    std::vector<std::uint32_t> routePointCounts;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<physics::RigidBodyId> routePointRigidBodyIds;
    std::vector<std::uint32_t> routePointRigidBodyIndices;
    std::vector<Diligent::float3> routePointLocalGuideOffsets;
};

struct ConstraintLayoutMapping
{
    /// Prepared host-side constraint-layout invalidation key produced by prepare().
    /// This describes when authored slot interpretation changes and is not the same as the live
    /// GPU custom-compute resource bindingGeneration exposed after uploadWorld().
    std::uint64_t layoutRevision = 0u;

    RigidParticleAttachmentConstraintLayoutMapping rigidParticleAttachments;
    RigidDistanceConstraintLayoutMapping rigidDistanceConstraints;
    RoutedCableConstraintLayoutMapping routedCables;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_CONSTRAINT_LAYOUT_MAPPING_H
