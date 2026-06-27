#ifndef CRESSIM_NEO_ENGINE_RENDER_SCENE_TYPES_H
#define CRESSIM_NEO_ENGINE_RENDER_SCENE_TYPES_H

#include <cstdint>

namespace cressim::neo::engine
{

struct EntityPoseMappingEntry
{
    std::uint32_t sourcePoseIndex = 0;
    std::uint32_t entityPoseIndex = 0;
    std::uint32_t reserved0       = 0;
    std::uint32_t reserved1       = 0;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RENDER_SCENE_TYPES_H
