#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H

#include "physics/physics_scene_gpu_state.h"
#include "physics/rigid_body_common.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"

#include <cstddef>
#include <cstdint>

namespace cressim::neo::physics
{

class PhysicsPassDispatcher
{
public:
    bool initialize(gpu::GpuDevice &device, std::uint32_t physicsContextId);

    bool clearCorrections(Diligent::IDeviceContext *computeContext,
                          PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants &constants);
    bool predictSoft(Diligent::IDeviceContext *computeContext,
                     const PhysicsSceneGpuState &sceneState, std::uint32_t softParticleCount,
                     const GpuSoftDispatchConstants &constants);
    bool updateRigidSurfaceWorldPositions(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState,
                                          std::uint32_t rigidSurfaceParticleCount,
                                          const GpuSoftDispatchConstants &constants);
    bool buildSoftBroadPhaseParticles(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      std::uint32_t totalParticleLikeCount,
                                      const GpuSoftDispatchConstants &constants);
    bool buildSoftBroadPhaseKeys(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t totalParticleLikeCount,
                                 const GpuSoftDispatchConstants &constants);
    bool sortSoftBroadPhase(Diligent::IDeviceContext *computeContext,
                            const PhysicsSceneGpuState &sceneState, std::uint32_t count);
    bool emitSoftCandidatePairs(Diligent::IDeviceContext *computeContext,
                                const PhysicsSceneGpuState &sceneState,
                                std::uint32_t softParticleCount,
                                const GpuSoftDispatchConstants &constants);
    bool generateSoftContacts(Diligent::IDeviceContext *computeContext,
                              const PhysicsSceneGpuState &sceneState,
                              const GpuSoftDispatchConstants &constants);
    bool clearSoftConstraintState(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState, std::uint32_t threadCount,
                                  const GpuSoftDispatchConstants &constants);
    bool solveSoftEdgeConstraints(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  std::uint32_t softEdgeCount,
                                  const GpuSoftDispatchConstants &constants);
    bool solveSoftTetConstraints(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState, std::uint32_t softTetCount,
                                 const GpuSoftDispatchConstants &constants);
    bool solveSoftContacts(Diligent::IDeviceContext *computeContext,
                           const PhysicsSceneGpuState &sceneState, std::uint32_t iterations,
                           const GpuSoftDispatchConstants &constants);
    bool updateSoftVelocities(Diligent::IDeviceContext *computeContext,
                              const PhysicsSceneGpuState &sceneState,
                              std::uint32_t softParticleCount,
                              const GpuSoftDispatchConstants &constants);
    bool predict(Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
                 std::uint32_t bodyCount, const GpuRigidDispatchConstants &constants);
    bool updateWorldAabbs(Diligent::IDeviceContext *computeContext,
                          const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants &constants);
    bool compactBroadPhaseBodySets(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                                   const GpuRigidDispatchConstants &constants);
    bool buildBroadPhase(Diligent::IDeviceContext *computeContext,
                         const PhysicsSceneGpuState &sceneState, std::uint32_t activeMovingCount,
                         const GpuRigidDispatchConstants &constants);
    bool finalizeBroadPhasePairs(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t activeMovingCount,
                                 const GpuRigidDispatchConstants &constants);
    bool emitBroadPhasePairs(Diligent::IDeviceContext *computeContext,
                             const PhysicsSceneGpuState &sceneState,
                             std::uint32_t activeMovingCount,
                             const GpuRigidDispatchConstants &constants);
    bool generateContacts(Diligent::IDeviceContext *computeContext,
                          const PhysicsSceneGpuState &sceneState, std::uint32_t pairCount);
    bool solveConstraints(Diligent::IDeviceContext *computeContext,
                          const PhysicsSceneGpuState &sceneState, std::uint32_t rigidBodyCount,
                          std::uint32_t pairCount, std::uint32_t iterations,
                          const GpuRigidDispatchConstants &constants);
    bool updateVelocities(Diligent::IDeviceContext *computeContext,
                          const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                          const GpuRigidDispatchConstants &constants);
    bool solveContactVelocities(Diligent::IDeviceContext *computeContext,
                                const PhysicsSceneGpuState &sceneState,
                                std::uint32_t rigidBodyCount, std::uint32_t pairCount,
                                std::uint32_t iterations,
                                const GpuRigidDispatchConstants &constants);

private:
    bool writeRigidDispatchConstants(Diligent::IDeviceContext *computeContext,
                                     const GpuRigidDispatchConstants &constants);
    bool writeSoftDispatchConstants(Diligent::IDeviceContext *computeContext,
                                    const GpuSoftDispatchConstants &constants);
    bool writeScanConstants(Diligent::IDeviceContext *computeContext,
                            const GpuPhysicsScanConstants &constants);
    bool writeRadixConstants(Diligent::IDeviceContext *computeContext,
                             const GpuPhysicsRadixConstants &constants);
    bool writeBroadPhaseBuildConstants(Diligent::IDeviceContext *computeContext,
                                       const GpuBroadPhaseBuildConstants &constants);
    bool writeBroadPhaseReductionConstants(Diligent::IDeviceContext *computeContext,
                                           const GpuBroadPhaseReductionConstants &constants);
    bool writeConstantsBuffer(Diligent::IDeviceContext *computeContext, Diligent::IBuffer *buffer,
                              const void *constants, std::size_t constantsSize);
    bool dispatchScanBlockPass(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, Diligent::IBuffer *input,
                               Diligent::IBuffer *output, Diligent::IBuffer *blockSums,
                               std::uint32_t count);
    bool dispatchScanAddOffsetsPass(Diligent::IDeviceContext *computeContext,
                                    Diligent::IBuffer *output,
                                    Diligent::IBuffer *scannedBlockOffsets, std::uint32_t count);
    bool dispatchExclusiveScanPass(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState, Diligent::IBuffer *input,
                                   Diligent::IBuffer *output, std::uint32_t count,
                                   std::uint32_t recursionLevel = 0u);
    bool dispatchReduceBroadPhaseExtentPass(Diligent::IDeviceContext *computeContext,
                                            const PhysicsSceneGpuState &sceneState,
                                            std::uint32_t bodyCount, bool useStaticSet);
    bool dispatchRadixSortPass(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t count,
                               bool useStaticSet);
    bool dispatchSoftRadixSortPass(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState, std::uint32_t count);
    bool dispatchGenerateContactsPass(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      std::uint32_t pairCount);
    bool dispatchSolveGatherPass(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState, std::uint32_t pairCount);
    bool dispatchSolveContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                            const PhysicsSceneGpuState &sceneState,
                                            std::uint32_t pairCount);

    gpu::ShaderLibrary mShaderLibrary{""};
    Diligent::Uint64 mPhysicsContextMask = 0;

    gpu::GpuComputePass mPredictPass;
    gpu::GpuComputePass mSoftPredictPass;
    gpu::GpuComputePass mUpdateRigidSurfaceWorldPositionsPass;
    gpu::GpuComputePass mBuildSoftBroadPhaseParticlesPass;
    gpu::GpuComputePass mBuildSoftBroadPhaseKeysPass;
    gpu::GpuComputePass mEmitSoftCandidatePairsPass;
    gpu::GpuComputePass mGenerateSoftContactsPass;
    gpu::GpuComputePass mClearSoftConstraintStatePass;
    gpu::GpuComputePass mSolveSoftEdgeConstraintsPass;
    gpu::GpuComputePass mSolveSoftTetConstraintsPass;
    gpu::GpuComputePass mSolveSoftContactsPass;
    gpu::GpuComputePass mApplySoftContactCorrectionsPass;
    gpu::GpuComputePass mUpdateSoftVelocitiesPass;
    gpu::GpuComputePass mUpdateWorldAabbsPass;
    gpu::GpuComputePass mScanBlockPass;
    gpu::GpuComputePass mScanAddOffsetsPass;
    gpu::GpuComputePass mCompactBodySetPass;
    gpu::GpuComputePass mFinalizeActiveBodiesPass;
    gpu::GpuComputePass mBuildBroadPhaseElementsPass;
    gpu::GpuComputePass mReduceExtentElementsPass;
    gpu::GpuComputePass mReduceExtentExtentsPass;
    gpu::GpuComputePass mMortonCodesPass;
    gpu::GpuComputePass mRadixClassifyPass;
    gpu::GpuComputePass mRadixFinalizePass;
    gpu::GpuComputePass mRadixScatterPass;
    gpu::GpuComputePass mBvhHierarchyPass;
    gpu::GpuComputePass mBvhBoundingBoxesPass;
    gpu::GpuComputePass mCountPairsPass;
    gpu::GpuComputePass mFinalizePairsPass;
    gpu::GpuComputePass mEmitPairsPass;
    gpu::GpuComputePass mBuildNarrowPhaseChunksPass;
    gpu::GpuComputePass mGenerateContactsPass;
    gpu::GpuComputePass mClearCorrectionsPass;
    gpu::GpuComputePass mSolveGatherPass;
    gpu::GpuComputePass mApplyCorrectionsPass;
    gpu::GpuComputePass mUpdateVelocitiesPass;
    gpu::GpuComputePass mSolveContactVelocitiesPass;
    gpu::GpuComputePass mApplyContactVelocitiesPass;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRigidDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mSoftDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mScanConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRadixConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseBuildConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseReductionConstantsBuffer;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
