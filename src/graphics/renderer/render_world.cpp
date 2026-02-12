#include "graphics/render_world.h"

#include <algorithm>

namespace cressim::neo::graphics
{

namespace
{

template <typename T>
void upsertByEntityId(std::vector<T>& entries, const T& value)
{
    const auto it = std::find_if(entries.begin(), entries.end(), [&](const T& entry) {
        return entry.entityId == value.entityId;
    });

    if (it == entries.end())
    {
        entries.push_back(value);
        return;
    }

    *it = value;
}

} // namespace

void RenderWorld::clear()
{
    mRenderables.clear();
    mCameras.clear();
    mDirectionalLights.clear();
}

void RenderWorld::upsertRenderable(const RenderableInstance& instance)
{
    upsertByEntityId(mRenderables, instance);
}

void RenderWorld::upsertCamera(const CameraData& camera)
{
    upsertByEntityId(mCameras, camera);
}

void RenderWorld::upsertDirectionalLight(const DirectionalLightData& light)
{
    upsertByEntityId(mDirectionalLights, light);
}

const std::vector<RenderableInstance>& RenderWorld::renderables() const noexcept
{
    return mRenderables;
}

const std::vector<CameraData>& RenderWorld::cameras() const noexcept
{
    return mCameras;
}

const std::vector<DirectionalLightData>& RenderWorld::directionalLights() const noexcept
{
    return mDirectionalLights;
}

} // namespace cressim::neo::graphics
