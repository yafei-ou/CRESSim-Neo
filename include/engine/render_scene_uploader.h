#ifndef CRESSIM_NEO_ENGINE_RENDER_SCENE_UPLOADER_H
#define CRESSIM_NEO_ENGINE_RENDER_SCENE_UPLOADER_H

#include "common/scene_primitives.h"
#include "engine/export.h"
#include "engine/render_scene_types.h"
#include "gpu/gpu_device.h"
#include "graphics/gpu_scene.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

class CRESSIM_NEO_ENGINE_API RenderSceneUploader
{
public:
    explicit RenderSceneUploader(gpu::GpuDevice &device);

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
    const common::SceneLayoutDesc &layout() const noexcept
    {
        return mLayout;
    }

private:
    bool ensureSharedPoseCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t entityCount);
    bool ensurePhysicsSyncCapacity(Diligent::IRenderDevice *renderDevice,
                                   std::uint32_t mappingCount);
    bool ensureRenderableCapacity(Diligent::IRenderDevice *renderDevice,
                                  std::uint32_t renderableCount);
    bool ensureCameraCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t cameraCount);
    bool ensureLightCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t lightCount);
    bool ensureLocalLightSelectionCapacity(Diligent::IRenderDevice *renderDevice,
                                           std::uint32_t selectionCount);
    bool writeBuffer(Diligent::IDeviceContext *computeContext, Diligent::IBuffer *buffer,
                     const void *data, std::size_t sizeBytes);

    gpu::GpuDevice &mDevice;
    common::SceneLayoutDesc mLayout{};
    bool mInitialized                            = false;
    std::uint32_t mPoseCapacity                  = 0;
    std::uint32_t mPhysicsSyncCapacity           = 0;
    std::uint32_t mEntityCount                   = 0;
    std::uint32_t mRenderableCapacity            = 0;
    std::uint32_t mRenderableCount               = 0;
    std::uint32_t mSoftBodyVertexBindingCapacity = 0;
    std::uint32_t mSoftBodyVertexBindingCount    = 0;
    std::uint32_t mCameraCapacity                = 0;
    std::uint32_t mCameraCount                   = 0;
    std::uint32_t mLightCapacity                 = 0;
    std::uint32_t mLightCount                    = 0;
    std::uint32_t mLocalLightSelectionCapacity   = 0;
    Diligent::Uint64 mGraphicsContextMask        = 0;
    Diligent::Uint64 mPhysicsContextMask         = 0;
    Diligent::Uint64 mSharedPoseContextMask      = 0;

    // Physics-only pose handoff resources.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mMappingBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;

    // Shared pose buffers: CPU-owned render poses and physics writeback both target these.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityPositionsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityOrientationsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityScalesBuffer;

    // Graphics-owned render scene resources.
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableMetadataBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableQueueInfoBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableVisibilityFlagsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableShadowCascadeMasksBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mSoftBodyVertexBindingBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mCameraInputsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPreparedCamerasBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mLightInputsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mLocalLightSelectionBuffer;
    std::uint64_t mPoseBindingGeneration                  = 1u;
    std::uint64_t mPhysicsSyncBindingGeneration           = 1u;
    std::uint64_t mSceneBindingGeneration                 = 1u;
    std::uint64_t mLastMappedSourcePoseBindingGeneration  = 0u;
    std::uint64_t mLastMappedOutputPoseBindingGeneration  = 0u;
    std::uint64_t mLastMappedPhysicsSyncBindingGeneration = 0u;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RENDER_SCENE_UPLOADER_H
