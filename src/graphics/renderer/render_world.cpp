#include "graphics/render_world.h"

#include <utility>

namespace cressim::neo::graphics
{

namespace
{

template <typename T>
void upsertByEntityId(std::vector<T>& entries,
                      std::unordered_map<common::EntityId, std::size_t>& indices, const T& value)
{
    const auto indexIt = indices.find(value.entityId);
    if (indexIt == indices.end())
    {
        const std::size_t newIndex = entries.size();
        entries.push_back(value);
        indices.emplace(value.entityId, newIndex);
        return;
    }

    entries[indexIt->second] = value;
}

template <typename T>
bool removeByEntityId(std::vector<T>& entries,
                      std::unordered_map<common::EntityId, std::size_t>& indices,
                      common::EntityId entityId)
{
    const auto indexIt = indices.find(entityId);
    if (indexIt == indices.end())
    {
        return false;
    }

    const std::size_t removedIndex = indexIt->second;
    const std::size_t lastIndex    = entries.size() - 1;
    if (removedIndex != lastIndex)
    {
        entries[removedIndex]                   = std::move(entries[lastIndex]);
        indices[entries[removedIndex].entityId] = removedIndex;
    }
    entries.pop_back();
    indices.erase(indexIt);
    return true;
}

} // namespace

void RenderWorld::clear()
{
    mRenderables.clear();
    mCameras.clear();
    mDirectionalLights.clear();

    mRenderableIndices.clear();
    mCameraIndices.clear();
    mDirectionalLightIndices.clear();
}

void RenderWorld::upsertRenderable(const RenderableInstance& instance)
{
    upsertByEntityId(mRenderables, mRenderableIndices, instance);
}

void RenderWorld::upsertCamera(const CameraData& camera)
{
    upsertByEntityId(mCameras, mCameraIndices, camera);
}

void RenderWorld::upsertDirectionalLight(const DirectionalLightData& light)
{
    upsertByEntityId(mDirectionalLights, mDirectionalLightIndices, light);
}

bool RenderWorld::removeRenderable(common::EntityId entityId)
{
    return removeByEntityId(mRenderables, mRenderableIndices, entityId);
}

bool RenderWorld::removeCamera(common::EntityId entityId)
{
    return removeByEntityId(mCameras, mCameraIndices, entityId);
}

bool RenderWorld::removeDirectionalLight(common::EntityId entityId)
{
    return removeByEntityId(mDirectionalLights, mDirectionalLightIndices, entityId);
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
