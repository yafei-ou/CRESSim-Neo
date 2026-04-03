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

    bool clearRigidCorrections(Diligent::IDeviceContext *computeContext,
                               PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool predictSoft(Diligent::IDeviceContext *computeContext,
                     const PhysicsSceneGpuState &sceneState, std::uint32_t softParticleCount,
                     const GpuSoftDispatchConstants &constants);
    bool updateRigidSurfaceWorldPositions(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState,
                                          std::uint32_t rigidSurfaceParticleCount,
                                          const GpuSoftDispatchConstants &constants);
    bool buildParticleBroadPhaseEntries(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t totalParticleLikeCount,
                                        const GpuSoftDispatchConstants &constants);
    bool buildParticleBroadPhaseKeys(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t totalParticleLikeCount,
                                     const GpuSoftDispatchConstants &constants);
    bool markParticleCellRangeStarts(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t totalParticleLikeCount,
                                     const GpuSoftDispatchConstants &constants);
    bool clearParticleCellRanges(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t cellRangeCapacity,
                                 const GpuSoftDispatchConstants &constants);
    bool buildParticleCellRanges(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t totalParticleLikeCount,
                                 const GpuSoftDispatchConstants &constants);
    bool sortParticleBroadPhase(Diligent::IDeviceContext *computeContext,
                                const PhysicsSceneGpuState &sceneState, std::uint32_t count);
    bool clearSoftNeighborMeta(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState);
    bool buildSoftSoftCandidatePairs(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t softParticleCount,
                                     const GpuSoftDispatchConstants &constants);
    bool buildSoftRigidCandidatePairs(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      std::uint32_t softParticleCount,
                                      const GpuSoftDispatchConstants &constants);
    bool generateSoftContacts(Diligent::IDeviceContext *computeContext,
                              const PhysicsSceneGpuState &sceneState, std::uint32_t candidateCount,
                              const GpuSoftDispatchConstants &constants);
    bool generateSoftRigidContacts(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState,
                                   std::uint32_t candidateCount,
                                   const GpuSoftDispatchConstants &constants);
    bool compactSoftContacts(Diligent::IDeviceContext *computeContext,
                             const PhysicsSceneGpuState &sceneState, std::uint32_t candidateCount,
                             const GpuSoftDispatchConstants &constants);
    bool compactSoftRigidContacts(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  std::uint32_t candidateCount,
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
                           const PhysicsSceneGpuState &sceneState, std::uint32_t contactCount,
                           const GpuSoftDispatchConstants &constants);
    bool solveSoftRigidContacts(Diligent::IDeviceContext *computeContext,
                                const PhysicsSceneGpuState &sceneState, std::uint32_t contactCount,
                                const GpuSoftDispatchConstants &constants);
    bool applySoftPositionCorrections(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      const GpuSoftDispatchConstants &constants);
    bool updateSoftVelocities(Diligent::IDeviceContext *computeContext,
                              const PhysicsSceneGpuState &sceneState,
                              std::uint32_t softParticleCount,
                              const GpuSoftDispatchConstants &constants);
    bool predictRigid(Diligent::IDeviceContext *computeContext,
                      const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                      const GpuRigidDispatchConstants &constants);
    bool updateRigidWorldAabbs(Diligent::IDeviceContext *computeContext,
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
    bool generateRigidContacts(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState,
                               std::uint32_t rigidPairCount);
    bool solveRigidContactConstraints(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      std::uint32_t rigidBodyCount, std::uint32_t rigidPairCount,
                                      const GpuRigidDispatchConstants &constants);
    bool applyRigidCorrections(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t rigidBodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool updateRigidVelocities(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool solveRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t rigidBodyCount, std::uint32_t rigidPairCount,
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
    bool dispatchGenerateRigidContactsPass(Diligent::IDeviceContext *computeContext,
                                           const PhysicsSceneGpuState &sceneState,
                                           std::uint32_t rigidPairCount);
    bool dispatchSolveRigidContactConstraintsPass(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState,
                                                  std::uint32_t rigidPairCount);
    bool dispatchSolveRigidContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState,
                                                 std::uint32_t rigidPairCount);

    gpu::ShaderLibrary mShaderLibrary{""};
    Diligent::Uint64 mPhysicsContextMask = 0;

    gpu::GpuComputePass mRigidPredictPass;
    gpu::GpuComputePass mSoftPredictPass;
    gpu::GpuComputePass mUpdateRigidSurfaceWorldPositionsPass;
    gpu::GpuComputePass mBuildParticleBroadPhaseEntriesPass;
    gpu::GpuComputePass mBuildParticleBroadPhaseKeysPass;
    gpu::GpuComputePass mMarkParticleCellRangeStartsPass;
    gpu::GpuComputePass mClearParticleCellRangesPass;
    gpu::GpuComputePass mBuildParticleCellRangesPass;
    gpu::GpuComputePass mCountSoftSoftCandidatePairsPass;
    gpu::GpuComputePass mFinalizeSoftSoftCandidatePairsPass;
    gpu::GpuComputePass mEmitSoftSoftCandidatePairsPass;
    gpu::GpuComputePass mCountSoftRigidCandidatePairsPass;
    gpu::GpuComputePass mFinalizeSoftRigidCandidatePairsPass;
    gpu::GpuComputePass mEmitSoftRigidCandidatePairsPass;
    gpu::GpuComputePass mGenerateSoftContactsPass;
    gpu::GpuComputePass mGenerateSoftRigidContactsPass;
    gpu::GpuComputePass mFinalizeActiveSoftContactsPass;
    gpu::GpuComputePass mCompactActiveSoftContactsPass;
    gpu::GpuComputePass mFinalizeActiveSoftRigidContactsPass;
    gpu::GpuComputePass mCompactActiveSoftRigidContactsPass;
    gpu::GpuComputePass mClearSoftConstraintStatePass;
    gpu::GpuComputePass mSolveSoftEdgeConstraintsPass;
    gpu::GpuComputePass mSolveSoftTetConstraintsPass;
    gpu::GpuComputePass mSolveSoftContactsPass;
    gpu::GpuComputePass mSolveSoftRigidContactsPass;
    gpu::GpuComputePass mApplySoftPositionCorrectionsPass;
    gpu::GpuComputePass mUpdateSoftVelocitiesPass;
    gpu::GpuComputePass mUpdateRigidWorldAabbsPass;
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
    gpu::GpuComputePass mGenerateRigidContactsPass;
    gpu::GpuComputePass mClearRigidCorrectionsPass;
    gpu::GpuComputePass mSolveRigidContactConstraintsPass;
    gpu::GpuComputePass mApplyRigidCorrectionsPass;
    gpu::GpuComputePass mUpdateRigidVelocitiesPass;
    gpu::GpuComputePass mSolveRigidContactVelocitiesPass;
    gpu::GpuComputePass mApplyRigidContactVelocitiesPass;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRigidDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mSoftDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mScanConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRadixConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseBuildConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseReductionConstantsBuffer;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
