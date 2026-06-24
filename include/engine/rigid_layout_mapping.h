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
    std::uint32_t rigidBodyCount    = 0u;
    std::uint32_t colliderCount     = 0u;
    std::uint64_t bindingGeneration = 0u;

    std::vector<common::EntityId> rigidBodyEntityIds;
    std::vector<std::uint32_t> rigidBodyEnvironmentIndices;

    std::vector<physics::ColliderId> colliderIds;
    std::vector<common::EntityId> colliderEntityIds;
    std::vector<std::uint32_t> colliderOwnerBodyIndices;
    std::vector<std::uint32_t> colliderEnvironmentIndices;

    std::vector<std::uint32_t> bodyColliderOffsets;
    std::vector<std::uint32_t> bodyColliderCounts;
    std::vector<std::uint32_t> bodyColliderIndices;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RIGID_LAYOUT_MAPPING_H
