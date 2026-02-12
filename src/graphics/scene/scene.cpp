#include "graphics/scene.h"

namespace cressim::neo::graphics
{

RenderResourceManager& Scene::resources() noexcept
{
    return mResources;
}

const RenderResourceManager& Scene::resources() const noexcept
{
    return mResources;
}

RenderWorld& Scene::world() noexcept
{
    return mWorld;
}

const RenderWorld& Scene::world() const noexcept
{
    return mWorld;
}

void Scene::clear()
{
    mWorld.clear();
}

} // namespace cressim::neo::graphics
