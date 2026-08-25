#ifndef CRESSIM_NEO_COMMON_ID_H
#define CRESSIM_NEO_COMMON_ID_H

#include <cstdint>

namespace cressim::neo::common
{

/// @brief Unique numeric identifier for entities within an ECS scene graph or World.
///
/// In Python, entity handles and identifiers are passed as native unsigned integers.
using EntityId                      = std::uint32_t;

/// @brief Sentinel constant representing an invalid or unallocated EntityId.
constexpr EntityId kInvalidEntityId = 0;

/// @brief Unique numeric identifier for managed engine resources (meshes, textures, materials).
using ResourceId                        = std::uint32_t;

/// @brief Sentinel constant representing an invalid or unallocated ResourceId.
constexpr ResourceId kInvalidResourceId = 0;

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_ID_H
