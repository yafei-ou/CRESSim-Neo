#ifndef CRESSIM_NEO_ENGINE_RENDER_SCENE_TYPES_H
#define CRESSIM_NEO_ENGINE_RENDER_SCENE_TYPES_H

#include <cstdint>

namespace cressim::neo::engine
{

/// @brief Maps one pose in a source pose buffer to one pose slot in the entity scene.
///
/// The two reserved fields are part of the GPU buffer layout and must be zero.
struct EntityPoseMappingEntry
{
    std::uint32_t sourcePoseIndex = 0; ///< Index into the source pose buffers.
    std::uint32_t entityPoseIndex = 0; ///< Destination entity-pose slot.
    std::uint32_t reserved0       = 0; ///< Reserved; set to zero.
    std::uint32_t reserved1       = 0; ///< Reserved; set to zero.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RENDER_SCENE_TYPES_H
