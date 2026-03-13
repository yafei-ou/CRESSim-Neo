#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H

#include "physics/physics_compute_pass.h"
#include "physics/physics_scene_gpu_state.h"
#include "physics/rigid_body_common.h"

#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#include <cstddef>
#include <cstdint>

namespace cressim::neo::physics
{

class PhysicsPassDispatcher
{
public:
    bool initialize(Diligent::IRenderDevice* renderDevice, std::uint32_t physicsContextId,
                    const char* shaderSourceDirectory);

    bool clearCorrections(Diligent::IDeviceContext* computeContext,
                          PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants& constants);
    bool predict(Diligent::IDeviceContext* computeContext, const PhysicsSceneGpuState& sceneState,
                 std::uint32_t bodyCount, const GpuRigidDispatchConstants& constants);
    bool updateWorldAabbs(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants& constants);
    bool compactActiveBodies(Diligent::IDeviceContext* computeContext,
                             const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                             const GpuRigidDispatchConstants& constants);
    bool buildBroadPhase(Diligent::IDeviceContext* computeContext,
                         const PhysicsSceneGpuState& sceneState, std::uint32_t activeMovingCount,
                         const GpuRigidDispatchConstants& constants);
    bool finalizeBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                                 const PhysicsSceneGpuState& sceneState,
                                 std::uint32_t activeMovingCount,
                                 const GpuRigidDispatchConstants& constants);
    bool emitBroadPhasePairs(Diligent::IDeviceContext* computeContext,
                             const PhysicsSceneGpuState& sceneState,
                             std::uint32_t activeMovingCount,
                             const GpuRigidDispatchConstants& constants);
    bool generateContacts(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t pairCount);
    bool solveConstraints(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t rigidBodyCount,
                          std::uint32_t pairCount, std::uint32_t iterations,
                          const GpuRigidDispatchConstants& constants);
    bool updateVelocities(Diligent::IDeviceContext* computeContext,
                          const PhysicsSceneGpuState& sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants& constants);

private:
    bool writeRigidDispatchConstants(Diligent::IDeviceContext* computeContext,
                                     const GpuRigidDispatchConstants& constants);
    bool writeScanConstants(Diligent::IDeviceContext* computeContext,
                            const GpuPhysicsScanConstants& constants);
    bool writeRadixConstants(Diligent::IDeviceContext* computeContext,
                             const GpuPhysicsRadixConstants& constants);
    bool writeConstantsBuffer(Diligent::IDeviceContext* computeContext, Diligent::IBuffer* buffer,
                              const void* constants, std::size_t constantsSize);
    bool dispatchScanBlockPass(Diligent::IDeviceContext* computeContext,
                               const PhysicsSceneGpuState& sceneState, Diligent::IBuffer* input,
                               Diligent::IBuffer* output, Diligent::IBuffer* blockSums,
                               std::uint32_t count);
    bool dispatchScanAddOffsetsPass(Diligent::IDeviceContext* computeContext,
                                    Diligent::IBuffer* output,
                                    Diligent::IBuffer* scannedBlockOffsets, std::uint32_t count);
    bool dispatchExclusiveScanPass(Diligent::IDeviceContext* computeContext,
                                   const PhysicsSceneGpuState& sceneState, Diligent::IBuffer* input,
                                   Diligent::IBuffer* output, std::uint32_t count,
                                   std::uint32_t recursionLevel = 0u);
    bool dispatchReduceBroadPhaseExtentPass(Diligent::IDeviceContext* computeContext,
                                            const PhysicsSceneGpuState& sceneState,
                                            std::uint32_t bodyCount, bool useStaticSet);
    bool dispatchRadixSortPass(Diligent::IDeviceContext* computeContext,
                               const PhysicsSceneGpuState& sceneState, std::uint32_t count,
                               bool useStaticSet);
    bool dispatchGenerateContactsPass(Diligent::IDeviceContext* computeContext,
                                      const PhysicsSceneGpuState& sceneState,
                                      std::uint32_t pairCount);
    bool dispatchSolveGatherPass(Diligent::IDeviceContext* computeContext,
                                 const PhysicsSceneGpuState& sceneState, std::uint32_t pairCount);

    gpu::ShaderLibrary mShaderLibrary{""};
    Diligent::Uint64 mPhysicsContextMask = 0;

    ComputePass mPredictPass;
    ComputePass mUpdateWorldAabbsPass;
    ComputePass mScanBlockPass;
    ComputePass mScanAddOffsetsPass;
    ComputePass mCompactActiveBodiesPass;
    ComputePass mFinalizeActiveBodiesPass;
    ComputePass mBuildBroadPhaseElementsPass;
    ComputePass mReduceExtentElementsPass;
    ComputePass mReduceExtentExtentsPass;
    ComputePass mMortonCodesPass;
    ComputePass mRadixClassifyPass;
    ComputePass mRadixFinalizePass;
    ComputePass mRadixScatterPass;
    ComputePass mBvhHierarchyPass;
    ComputePass mBvhBoundingBoxesPass;
    ComputePass mCountPairsPass;
    ComputePass mFinalizePairsPass;
    ComputePass mEmitPairsPass;
    ComputePass mBuildNarrowPhaseChunksPass;
    ComputePass mGenerateContactsPass;
    ComputePass mClearCorrectionsPass;
    ComputePass mSolveGatherPass;
    ComputePass mApplyCorrectionsPass;
    ComputePass mUpdateVelocitiesPass;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRigidDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mScanConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRadixConstantsBuffer;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
