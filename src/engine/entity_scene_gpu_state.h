#ifndef CRESSIM_NEO_SRC_ENGINE_ENTITY_SCENE_GPU_STATE_H
#define CRESSIM_NEO_SRC_ENGINE_ENTITY_SCENE_GPU_STATE_H

#include "engine/render_scene_types.h"
#include "gpu/gpu_compute_pass.h"
#include "gpu/gpu_device.h"

#include "common/scene_primitives.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"

#include <cstdint>
#include <vector>

namespace cressim::neo::engine
{

class EntitySceneGpuState
{
public:
    explicit EntitySceneGpuState(gpu::GpuDevice &device);

    bool initialize();
    void shutdown();

    bool uploadAuthoredEntityPoses(const std::vector<Diligent::float4> &positions,
                                   const std::vector<Diligent::float4> &orientations,
                                   const std::vector<Diligent::float4> &scales);
    bool applyMappedEntityPoses(const common::PoseBufferView &sourcePoses,
                                const std::vector<EntityPoseMappingEntry> &mappings);

    common::PoseBufferView poseView() const noexcept;
    std::uint32_t entityCount() const noexcept;

private:
    bool ensureSharedPoseCapacity(Diligent::IRenderDevice *renderDevice, std::uint32_t entityCount);
    bool ensurePhysicsSyncCapacity(Diligent::IRenderDevice *renderDevice,
                                   std::uint32_t mappingCount);
    bool writeBuffer(Diligent::IDeviceContext *computeContext, Diligent::IBuffer *buffer,
                     const void *data, std::size_t sizeBytes);

    gpu::GpuDevice &mDevice;
    bool mInitialized                       = false;
    std::uint32_t mPoseCapacity             = 0;
    std::uint32_t mPhysicsSyncCapacity      = 0;
    std::uint32_t mEntityCount              = 0;
    Diligent::Uint64 mGraphicsContextMask   = 0;
    Diligent::Uint64 mPhysicsContextMask    = 0;
    Diligent::Uint64 mSharedPoseContextMask = 0;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mMappingBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityPositionsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityOrientationsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEntityScalesBuffer;
    gpu::GpuComputePass mEntityPoseSyncPass;
    std::uint64_t mPoseBindingGeneration                  = 1u;
    std::uint64_t mPhysicsSyncBindingGeneration           = 1u;
    std::uint64_t mLastMappedSourcePoseBindingGeneration  = 0u;
    std::uint64_t mLastMappedOutputPoseBindingGeneration  = 0u;
    std::uint64_t mLastMappedPhysicsSyncBindingGeneration = 0u;
};

} // namespace cressim::neo::engine

#endif
