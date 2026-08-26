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
    /// @brief Creates an uploader associated with @p device. Call initialize() before use.
    explicit RenderSceneUploader(gpu::GpuDevice &device);

    /// @brief Releases uploader resources.
    ~RenderSceneUploader();

    /// @brief Initializes GPU scene buffers and the physics-to-render pose synchronization pass.
    /// @param layout Per-environment scene capacities used to size GPU buffers.
    /// @return False when compatible graphics and physics backend contexts or shader resources are
    /// unavailable.
    bool initialize(const common::SceneLayoutDesc &layout = common::SceneLayoutDesc{});

    /// @brief Releases all GPU resources and resets this uploader to its uninitialized state.
    void shutdown();

    /// @brief Uploads complete entity pose arrays to the shared entity-scene pose buffers.
    /// @return False unless initialized, all three arrays have equal length, and the physics
    /// backend is available.
    bool uploadEntityPoseData(const std::vector<Diligent::float4> &positions,
                              const std::vector<Diligent::float4> &orientations,
                              const std::vector<Diligent::float4> &scales);

    /// @brief Dispatches a GPU copy of selected source poses into entity-scene pose slots.
    /// @param sourcePoses Source GPU pose buffers. All three buffers and a non-zero count are
    /// required when @p mappings is non-empty.
    /// @param mappings Source-to-destination pose mappings to apply.
    /// @return False when uninitialized, required buffers/backend resources are unavailable, or
    /// the synchronization dispatch cannot be submitted.
    bool applyMappedEntityPoses(const common::PoseBufferView &sourcePoses,
                                const std::vector<EntityPoseMappingEntry> &mappings);

    /// @brief Uploads renderable metadata; an empty vector clears the recorded renderable count.
    bool uploadRenderableMetadata(const std::vector<graphics::GpuRenderableMetadata> &renderables);

    /// @brief Uploads render-queue information for renderable slots.
    bool uploadRenderableQueueInfo(const std::vector<graphics::GpuRenderableQueueInfo> &queueInfo);

    /// @brief Uploads soft-body vertex-to-particle bindings.
    bool uploadSoftBodyVertexBindings(
        const std::vector<graphics::GpuSoftBodyVertexBinding> &bindings);

    /// @brief Uploads camera inputs; an empty vector clears the recorded camera count.
    bool uploadCameraInputs(const std::vector<graphics::GpuCameraInput> &cameras);

    /// @brief Uploads light inputs; an empty vector clears the recorded light count.
    bool uploadLightInputs(const std::vector<graphics::GpuLightInput> &lights);

    /// @brief Uploads per-renderable local-light selections.
    bool uploadLocalLightSelections(
        const std::vector<graphics::GpuLocalLightSelection> &selections);

    /// @brief Returns a non-owning view of the internally managed GPU entity scene.
    graphics::GpuEntitySceneView sceneView() const noexcept;

    /// @brief Returns a scene view that uses caller-supplied pose buffers with this uploader's
    /// uploaded renderable, camera, and light buffers.
    graphics::GpuEntitySceneView sceneView(const common::PoseBufferView &poses,
                                           std::uint32_t entityCount) const noexcept;

    /// @brief Returns the layout supplied to initialize(), or the default layout after shutdown.
    const common::SceneLayoutDesc &layout() const noexcept;

private:
    struct CRESSIM_NEO_LOCAL Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RENDER_SCENE_UPLOADER_H
