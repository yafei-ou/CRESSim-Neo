#ifndef CRESSIM_NEO_GPU_GPU_SCENE_SYNC_H
#define CRESSIM_NEO_GPU_GPU_SCENE_SYNC_H

#include "gpu/export.h"
#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"

#include <cstdint>
#include <vector>

namespace cressim::neo::gpu
{

class CRESSIM_NEO_GPU_API GpuSceneSync
{
public:
    explicit GpuSceneSync(GpuDevice &device);

    bool initialize(const GpuSceneLayoutDesc &layout = GpuSceneLayoutDesc{});
    void shutdown();

    bool syncEntityPoseData(const std::vector<Diligent::float4> &positions,
                            const std::vector<Diligent::float4> &orientations,
                            const std::vector<Diligent::float4> &scales);
    bool syncEntityPoses(const GpuPoseBufferView &sourcePoses,
                         const std::vector<GpuEntityPoseMappingEntry> &mappings);
    bool syncRenderableMetadata(const std::vector<GpuRenderableMetadata> &renderables);
    bool syncRenderableQueueInfo(const std::vector<GpuRenderableQueueInfo> &queueInfo);
    bool syncCameraInputs(const std::vector<GpuCameraInput> &cameras);
    bool syncLightInputs(const std::vector<GpuLightInput> &lights);
    bool syncLocalLightSelections(const std::vector<GpuLocalLightSelection> &selections);

    GpuEntitySceneView sceneView() const noexcept;
    const GpuSceneLayoutDesc &layout() const noexcept
    {
        return mLayout;
    }

private:
    bool ensureCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t entityCount,
                        std::uint32_t contextId);
    bool writeBuffer(Diligent::IDeviceContext *computeContext, Diligent::IBuffer *buffer,
                     const void *data, std::size_t sizeBytes);

    GpuDevice &mDevice;
    GpuSceneLayoutDesc mLayout{};
    bool mInitialized                 = false;
    std::uint32_t mCapacity           = 0;
    std::uint32_t mEntityCount        = 0;
    std::uint32_t mRenderableCapacity = 0;
    std::uint32_t mRenderableCount    = 0;
    std::uint32_t mCameraCapacity     = 0;
    std::uint32_t mCameraCount        = 0;
    std::uint32_t mLightCapacity      = 0;
    std::uint32_t mLightCount         = 0;
    Diligent::Uint64 mContextMask     = 0;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> mMappingBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityPositionsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityOrientationsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityScalesBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableMetadataBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableQueueInfoBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableVisibilityFlagsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRenderableShadowCascadeMasksBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mCameraInputsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPreparedCamerasBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mLightInputsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mLocalLightSelectionBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_SCENE_SYNC_H
