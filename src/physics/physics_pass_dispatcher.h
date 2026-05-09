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
                     const PhysicsSceneGpuState &sceneState, std::uint32_t particleCount,
                     const GpuParticleDispatchConstants &constants);
    bool buildParticleBroadPhaseEntries(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t totalParticleLikeCount,
                                        const GpuParticleDispatchConstants &constants);
    bool buildParticleBroadPhaseKeys(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t totalParticleLikeCount,
                                     const GpuParticleDispatchConstants &constants);
    bool markParticleCellRangeStarts(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t totalParticleLikeCount,
                                     const GpuParticleDispatchConstants &constants);
    bool clearParticleCellRanges(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t cellRangeCapacity,
                                 const GpuParticleDispatchConstants &constants);
    bool buildParticleCellRanges(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t totalParticleLikeCount,
                                 const GpuParticleDispatchConstants &constants);
    bool sortParticleBroadPhase(Diligent::IDeviceContext *computeContext,
                                const PhysicsSceneGpuState &sceneState, std::uint32_t count);
    bool clearParticleNeighborMeta(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState);
    bool buildParticleParticleCandidatePairs(Diligent::IDeviceContext *computeContext,
                                             const PhysicsSceneGpuState &sceneState,
                                             std::uint32_t particleCount,
                                             const GpuParticleDispatchConstants &constants);
    bool buildParticleRigidCandidatePairs(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState,
                                          std::uint32_t particleCount,
                                          const GpuParticleDispatchConstants &constants);
    bool prepareParticleCandidateIndirectArgs(Diligent::IDeviceContext *computeContext,
                                              const PhysicsSceneGpuState &sceneState);
    bool prepareParticleActiveIndirectArgs(Diligent::IDeviceContext *computeContext,
                                           const PhysicsSceneGpuState &sceneState);
    bool generateParticleExplicitContacts(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState,
                                          const GpuParticleDispatchConstants &constants);
    bool generateParticleRigidContacts(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       const GpuParticleDispatchConstants &constants);
    bool compactParticleExplicitContacts(Diligent::IDeviceContext *computeContext,
                                         const PhysicsSceneGpuState &sceneState,
                                         const GpuParticleDispatchConstants &constants);
    bool compactParticleRigidContacts(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      const GpuParticleDispatchConstants &constants);
    bool clearSoftConstraintState(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState, std::uint32_t threadCount,
                                  const GpuParticleDispatchConstants &constants);
    bool solveSoftEdgeConstraints(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  std::uint32_t softEdgeCount,
                                  const GpuParticleDispatchConstants &constants);
    bool solveSoftTetConstraints(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState, std::uint32_t softTetCount,
                                 const GpuParticleDispatchConstants &constants);
    bool applySoftEdgeCorrections(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  const GpuParticleDispatchConstants &constants);
    bool applySoftTetCorrections(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 const GpuParticleDispatchConstants &constants);
    bool solveParticleExplicitContacts(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       const GpuParticleDispatchConstants &constants);
    bool solveParticleRigidContacts(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState,
                                    const GpuParticleDispatchConstants &constants);
    bool applyParticlePositionCorrections(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState,
                                          const GpuParticleDispatchConstants &constants);
    bool updateParticleVelocities(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  std::uint32_t particleCount,
                                  const GpuParticleDispatchConstants &constants);
    bool computeFluidDensityLambda(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState,
                                   const GpuParticleDispatchConstants &constants);
    bool computeFluidDeltaPositions(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState,
                                    const GpuParticleDispatchConstants &constants);
    bool applyFluidDeltaPositions(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  const GpuParticleDispatchConstants &constants);
    bool applyFluidXsphViscosity(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 const GpuParticleDispatchConstants &constants);
    bool solveParticleContactVelocities(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t particleCount,
                                        std::uint32_t iterations);
    bool solveParticleRigidContactVelocities(
        Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
        std::uint32_t particleCount, std::uint32_t rigidBodyCount,
        std::uint32_t iterations, const GpuRigidDispatchConstants &rigidConstants);
    bool updateSoftTriangleNormals(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState,
                                   std::uint32_t renderTriangleCount);
    bool updateSoftRenderNormals(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t renderVertexCount);
    bool updateSoftBodyBounds(Diligent::IDeviceContext *computeContext,
                              const PhysicsSceneGpuState &sceneState, std::uint32_t softBodyCount,
                              std::uint32_t softBodyBoundsChunkCount);
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
    bool prepareRigidIndirectArgs(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState);
    bool generateRigidContacts(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState);
    bool solveRigidContactConstraints(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      std::uint32_t rigidBodyCount,
                                      const GpuRigidDispatchConstants &constants);
    bool solveBallJointConstraints(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState,
                                   const GpuRigidDispatchConstants &constants);
    bool solveHingeJointConstraints(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState,
                                    const GpuRigidDispatchConstants &constants);
    bool solveSliderJointConstraints(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     const GpuRigidDispatchConstants &constants);
    bool solveHingeJointTargetVelocities(Diligent::IDeviceContext *computeContext,
                                         const PhysicsSceneGpuState &sceneState,
                                         const GpuRigidDispatchConstants &constants);
    bool solveSliderJointTargetVelocities(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState,
                                          const GpuRigidDispatchConstants &constants);
    bool applyRigidVelocityCorrections(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       std::uint32_t rigidBodyCount,
                                       const GpuRigidDispatchConstants &constants);
    bool applyRigidCorrections(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t rigidBodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool updateRigidVelocities(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool solveRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t rigidBodyCount, std::uint32_t iterations,
                                     const GpuRigidDispatchConstants &constants);
    bool updateRigidDispatchConstants(Diligent::IDeviceContext *computeContext,
                                      const GpuRigidDispatchConstants &constants);
    bool recreateSceneBindingVariants();

private:
    bool writeRigidDispatchConstants(Diligent::IDeviceContext *computeContext,
                                     const GpuRigidDispatchConstants &constants);
    bool writeRigidJointDispatchConstants(Diligent::IDeviceContext *computeContext,
                                          const GpuRigidJointDispatchConstants &constants);
    bool writeParticleDispatchConstants(Diligent::IDeviceContext *computeContext,
                                        const GpuParticleDispatchConstants &constants);
    bool writeSoftRenderDispatchConstants(Diligent::IDeviceContext *computeContext,
                                          const GpuSoftRenderDispatchConstants &constants);
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
                                           const PhysicsSceneGpuState &sceneState);
    bool dispatchSolveRigidContactConstraintsPass(Diligent::IDeviceContext *computeContext,
                                                  const PhysicsSceneGpuState &sceneState);
    bool dispatchSolveBallJointConstraintsPass(Diligent::IDeviceContext *computeContext,
                                               const PhysicsSceneGpuState &sceneState,
                                               std::uint32_t jointCount);
    bool dispatchSolveHingeJointConstraintsPass(Diligent::IDeviceContext *computeContext,
                                                gpu::GpuComputePass &pass,
                                                const PhysicsSceneGpuState &sceneState,
                                                Diligent::IBuffer *jointIndicesBuffer,
                                                std::uint32_t jointCount);
    bool dispatchSolveSliderJointConstraintsPass(Diligent::IDeviceContext *computeContext,
                                                 gpu::GpuComputePass &pass,
                                                 const PhysicsSceneGpuState &sceneState,
                                                 Diligent::IBuffer *jointIndicesBuffer,
                                                 std::uint32_t jointCount);
    bool dispatchSolveHingeJointVelocityTargetsPass(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState,
                                                    Diligent::IBuffer *jointIndicesBuffer,
                                                    std::uint32_t jointCount);
    bool dispatchSolveSliderJointVelocityTargetsPass(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     Diligent::IBuffer *jointIndicesBuffer,
                                                     std::uint32_t jointCount);
    bool dispatchApplyRigidVelocityCorrectionsPass(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t rigidBodyCount);
    bool dispatchApplyRigidCorrectionsPass(Diligent::IDeviceContext *computeContext,
                                           const PhysicsSceneGpuState &sceneState,
                                           std::uint32_t rigidBodyCount);
    bool dispatchSolveParticleContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState);
    bool dispatchSolveParticleRigidContactVelocitiesPass(
        Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState);
    bool dispatchSolveRigidContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState);

    gpu::ShaderLibrary mShaderLibrary{""};
    Diligent::Uint64 mPhysicsContextMask = 0;

    gpu::GpuComputePass mRigidPredictPass;
    gpu::GpuComputePass mSoftPredictPass;
    gpu::GpuComputePass mBuildParticleBroadPhaseEntriesPass;
    gpu::GpuComputePass mBuildParticleBroadPhaseKeysPass;
    gpu::GpuComputePass mMarkParticleCellRangeStartsPass;
    gpu::GpuComputePass mClearParticleCellRangesPass;
    gpu::GpuComputePass mBuildParticleCellRangesPass;
    gpu::GpuComputePass mCountParticleParticleCandidatePairsPass;
    gpu::GpuComputePass mFinalizeParticleParticleCandidatePairsPass;
    gpu::GpuComputePass mEmitParticleParticleCandidatePairsPass;
    gpu::GpuComputePass mCountParticleRigidCandidatePairsPass;
    gpu::GpuComputePass mFinalizeParticleRigidCandidatePairsPass;
    gpu::GpuComputePass mEmitParticleRigidCandidatePairsPass;
    gpu::GpuComputePass mGenerateParticleExplicitContactsPass;
    gpu::GpuComputePass mGenerateParticleRigidContactsPass;
    gpu::GpuComputePass mPrepareParticleCandidateIndirectArgsPass;
    gpu::GpuComputePass mPrepareParticleActiveIndirectArgsPass;
    gpu::GpuComputePass mFinalizeActiveParticleExplicitContactsPass;
    gpu::GpuComputePass mCompactActiveParticleExplicitContactsPass;
    gpu::GpuComputePass mFinalizeActiveParticleRigidContactsPass;
    gpu::GpuComputePass mCompactActiveParticleRigidContactsPass;
    gpu::GpuComputePass mClearSoftConstraintStatePass;
    gpu::GpuComputePass mSolveSoftEdgeConstraintsPass;
    gpu::GpuComputePass mSolveSoftTetConstraintsPass;
    gpu::GpuComputePass mApplySoftEdgeCorrectionsPass;
    gpu::GpuComputePass mApplySoftTetCorrectionsPass;
    gpu::GpuComputePass mSolveParticleExplicitContactsPass;
    gpu::GpuComputePass mSolveParticleRigidContactsPass;
    gpu::GpuComputePass mApplyParticlePositionCorrectionsPass;
    gpu::GpuComputePass mUpdateParticleVelocitiesPass;
    gpu::GpuComputePass mComputeFluidDensityLambdaPass;
    gpu::GpuComputePass mComputeFluidDeltaPositionsPass;
    gpu::GpuComputePass mApplyFluidDeltaPositionsPass;
    gpu::GpuComputePass mApplyFluidXsphViscosityPass;
    gpu::GpuComputePass mSolveParticleContactVelocitiesPass;
    gpu::GpuComputePass mSolveParticleRigidContactVelocitiesPass;
    gpu::GpuComputePass mApplyParticleContactVelocitiesPass;
    gpu::GpuComputePass mUpdateSoftTriangleNormalsPass;
    gpu::GpuComputePass mUpdateSoftRenderNormalsPass;
    gpu::GpuComputePass mUpdateSoftBodyBoundsPass;
    gpu::GpuComputePass mFinalizeSoftBodyBoundsPass;
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
    gpu::GpuComputePass mPrepareRigidIndirectArgsPass;
    gpu::GpuComputePass mGenerateRigidContactsPass;
    gpu::GpuComputePass mClearRigidCorrectionsPass;
    gpu::GpuComputePass mSolveRigidContactConstraintsPass;
    gpu::GpuComputePass mSolveBallJointConstraintsPass;
    gpu::GpuComputePass mSolveHingeJointConstraintsPassivePass;
    gpu::GpuComputePass mSolveHingeJointConstraintsTargetPositionPass;
    gpu::GpuComputePass mSolveSliderJointConstraintsPassivePass;
    gpu::GpuComputePass mSolveSliderJointConstraintsTargetPositionPass;
    gpu::GpuComputePass mSolveHingeJointTargetVelocitiesPass;
    gpu::GpuComputePass mSolveSliderJointTargetVelocitiesPass;
    gpu::GpuComputePass mApplyRigidCorrectionsPass;
    gpu::GpuComputePass mUpdateRigidVelocitiesPass;
    gpu::GpuComputePass mSolveRigidContactVelocitiesPass;
    gpu::GpuComputePass mApplyRigidContactVelocitiesPass;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRigidDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRigidJointDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mParticleDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mSoftRenderDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mScanConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRadixConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseBuildConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseReductionConstantsBuffer;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
