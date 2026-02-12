#ifndef CRESSIM_NEO_ENGINE_WORLD_TO_RENDER_WORLD_SYNC_H
#define CRESSIM_NEO_ENGINE_WORLD_TO_RENDER_WORLD_SYNC_H

#include "engine/world.h"
#include "graphics/render_world.h"

namespace cressim::neo::engine::detail
{

void syncWorldToRenderWorld(const World& world, graphics::RenderWorld& renderWorld);

} // namespace cressim::neo::engine::detail

#endif // CRESSIM_NEO_ENGINE_WORLD_TO_RENDER_WORLD_SYNC_H
