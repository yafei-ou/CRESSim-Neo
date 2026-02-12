#ifndef CRESSIM_NEO_GRAPHICS_SCENE_H
#define CRESSIM_NEO_GRAPHICS_SCENE_H

#include "graphics/export.h"
#include "graphics/render_resource_manager.h"
#include "graphics/render_world.h"

namespace cressim::neo::graphics
{

class CRESSIM_NEO_GRAPHICS_API Scene
{
public:
    RenderResourceManager& resources() noexcept;
    const RenderResourceManager& resources() const noexcept;

    RenderWorld& world() noexcept;
    const RenderWorld& world() const noexcept;

    void clear();

private:
    RenderResourceManager mResources;
    RenderWorld mWorld;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_SCENE_H
