#ifndef CRESSIM_NEO_ENGINE_RENDER_SCENE_UPLOADER_H
#define CRESSIM_NEO_ENGINE_RENDER_SCENE_UPLOADER_H

#include "common/scene_primitives.h"
#include "engine/export.h"
#include "engine/render_scene_types.h"
#include "graphics/gpu_scene.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace cressim::neo::gpu
{
class GpuDevice;
}

namespace cressim::neo::engine
{

class CRESSIM_NEO_ENGINE_API RenderSceneUploader
{
public:
    explicit RenderSceneUploader(gpu::GpuDevice &device);
    ~RenderSceneUploader();

    bool initialize(const common::SceneLayoutDesc &layout = common::SceneLayoutDesc{});
    void shutdown();

    bool uploadEntityPoseData(const std::vector<Diligent::float4> &positions,
                              const std::vector<Diligent::float4> &orientations,
                              const std::vector<Diligent::float4> &scales);
    bool applyMappedEntityPoses(const common::PoseBufferView &sourcePoses,
                                const std::vector<EntityPoseMappingEntry> &mappings);
    bool uploadRenderableMetadata(const std::vector<graphics::GpuRenderableMetadata> &renderables);
    bool uploadRenderableQueueInfo(const std::vector<graphics::GpuRenderableQueueInfo> &queueInfo);
    bool uploadSoftBodyVertexBindings(
        const std::vector<graphics::GpuSoftBodyVertexBinding> &bindings);
    bool uploadCameraInputs(const std::vector<graphics::GpuCameraInput> &cameras);
    bool uploadLightInputs(const std::vector<graphics::GpuLightInput> &lights);
    bool uploadLocalLightSelections(
        const std::vector<graphics::GpuLocalLightSelection> &selections);

    graphics::GpuEntitySceneView sceneView() const noexcept;
    graphics::GpuEntitySceneView sceneView(const common::PoseBufferView &poses,
                                           std::uint32_t entityCount) const noexcept;
    const common::SceneLayoutDesc &layout() const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RENDER_SCENE_UPLOADER_H
