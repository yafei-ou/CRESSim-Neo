#include "graphics/render_world.h"

#include <utility>

namespace cressim::neo::graphics
{

namespace
{

constexpr std::uint32_t kInvalidSlot = 0xffffffffu;

std::uint32_t allocateDenseSlot(
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& freeSlotsByEnv,
    std::unordered_map<std::uint32_t, std::uint32_t>& nextSlotByEnv, std::uint32_t envIndex)
{
    auto& freeSlots = freeSlotsByEnv[envIndex];
    if (!freeSlots.empty())
    {
        const std::uint32_t slot = freeSlots.back();
        freeSlots.pop_back();
        return slot;
    }

    std::uint32_t& nextSlot = nextSlotByEnv[envIndex];
    return nextSlot++;
}

void reclaimDenseSlot(std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& freeSlotsByEnv,
                      std::uint32_t envIndex, std::uint32_t slot)
{
    if (slot == kInvalidSlot)
    {
        return;
    }
    freeSlotsByEnv[envIndex].push_back(slot);
}

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
    mGpuEntityScene = {};
    mGpuEntityPoseIndices.clear();

    mRenderableIndices.clear();
    mCameraIndices.clear();
    mDirectionalLightIndices.clear();
    mNextRenderableSlotByEnv.clear();
    mNextCameraSlotByEnv.clear();
    mNextDirectionalLightSlotByEnv.clear();
    mFreeRenderableSlotsByEnv.clear();
    mFreeCameraSlotsByEnv.clear();
    mFreeDirectionalLightSlotsByEnv.clear();
}

void RenderWorld::upsertRenderable(const RenderableInstance& instance)
{
    RenderableInstance assigned = instance;
    const auto indexIt          = mRenderableIndices.find(instance.entityId);
    if (indexIt == mRenderableIndices.end())
    {
        if (assigned.objectSlot == kInvalidSlot)
        {
            assigned.objectSlot = allocateDenseSlot(mFreeRenderableSlotsByEnv,
                                                    mNextRenderableSlotByEnv, assigned.envIndex);
        }
    }
    else
    {
        assigned.objectSlot = mRenderables[indexIt->second].objectSlot;
    }
    upsertByEntityId(mRenderables, mRenderableIndices, assigned);
}

void RenderWorld::upsertCamera(const CameraData& camera)
{
    CameraData assigned = camera;
    const auto indexIt  = mCameraIndices.find(camera.entityId);
    if (indexIt == mCameraIndices.end())
    {
        if (assigned.cameraSlot == kInvalidSlot)
        {
            assigned.cameraSlot =
                allocateDenseSlot(mFreeCameraSlotsByEnv, mNextCameraSlotByEnv, assigned.envIndex);
        }
    }
    else
    {
        assigned.cameraSlot = mCameras[indexIt->second].cameraSlot;
    }
    upsertByEntityId(mCameras, mCameraIndices, assigned);
}

void RenderWorld::upsertDirectionalLight(const DirectionalLightData& light)
{
    DirectionalLightData assigned = light;
    const auto indexIt            = mDirectionalLightIndices.find(light.entityId);
    if (indexIt == mDirectionalLightIndices.end())
    {
        if (assigned.lightSlot == kInvalidSlot)
        {
            assigned.lightSlot = allocateDenseSlot(
                mFreeDirectionalLightSlotsByEnv, mNextDirectionalLightSlotByEnv, assigned.envIndex);
        }
    }
    else
    {
        assigned.lightSlot = mDirectionalLights[indexIt->second].lightSlot;
    }
    upsertByEntityId(mDirectionalLights, mDirectionalLightIndices, assigned);
}

bool RenderWorld::removeRenderable(common::EntityId entityId)
{
    const auto indexIt = mRenderableIndices.find(entityId);
    if (indexIt != mRenderableIndices.end())
    {
        const RenderableInstance& instance = mRenderables[indexIt->second];
        reclaimDenseSlot(mFreeRenderableSlotsByEnv, instance.envIndex, instance.objectSlot);
    }
    return removeByEntityId(mRenderables, mRenderableIndices, entityId);
}

bool RenderWorld::removeCamera(common::EntityId entityId)
{
    const auto indexIt = mCameraIndices.find(entityId);
    if (indexIt != mCameraIndices.end())
    {
        const CameraData& camera = mCameras[indexIt->second];
        reclaimDenseSlot(mFreeCameraSlotsByEnv, camera.envIndex, camera.cameraSlot);
    }
    return removeByEntityId(mCameras, mCameraIndices, entityId);
}

bool RenderWorld::removeDirectionalLight(common::EntityId entityId)
{
    const auto indexIt = mDirectionalLightIndices.find(entityId);
    if (indexIt != mDirectionalLightIndices.end())
    {
        const DirectionalLightData& light = mDirectionalLights[indexIt->second];
        reclaimDenseSlot(mFreeDirectionalLightSlotsByEnv, light.envIndex, light.lightSlot);
    }
    return removeByEntityId(mDirectionalLights, mDirectionalLightIndices, entityId);
}

void RenderWorld::setGpuEntityScene(
    const gpu::GpuEntitySceneView& sceneView,
    const std::unordered_map<common::EntityId, std::uint32_t>& poseIndices) noexcept
{
    mGpuEntityScene       = sceneView;
    mGpuEntityPoseIndices = poseIndices;
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

const gpu::GpuEntitySceneView& RenderWorld::gpuEntityScene() const noexcept
{
    return mGpuEntityScene;
}

const std::unordered_map<common::EntityId, std::uint32_t>& RenderWorld::gpuEntityPoseIndices()
    const noexcept
{
    return mGpuEntityPoseIndices;
}

} // namespace cressim::neo::graphics
