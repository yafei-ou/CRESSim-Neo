#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H

#include "physics/physics_scene_gpu_state.h"
#include "physics/rigid_body_common.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_source_provider.h"

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
    bool updateSolverConfig(Diligent::IDeviceContext *computeContext,
                            const GpuPhysicsSolverConfig &config);

    bool clearRigidCorrections(Diligent::IDeviceContext *computeContext,
                               PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool updateHingeJointRuntimeState(Diligent::IDeviceContext *computeContext,
                                      const PhysicsSceneGpuState &sceneState,
                                      std::uint32_t jointCount);
    bool updateSliderJointRuntimeState(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       std::uint32_t jointCount);
    bool clearHingeJointConstraintState(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t jointCount);
    bool clearSphericalJointConstraintState(Diligent::IDeviceContext *computeContext,
                                            const PhysicsSceneGpuState &sceneState,
                                            std::uint32_t jointCount);
    bool clearSliderJointConstraintState(Diligent::IDeviceContext *computeContext,
                                         const PhysicsSceneGpuState &sceneState,
                                         std::uint32_t jointCount);
    bool predictSoft(Diligent::IDeviceContext *computeContext,
                     const PhysicsSceneGpuState &sceneState, std::uint32_t particleCount,
                     const GpuParticleDispatchConstants &constants);
    bool syncRigidProxyParticles(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t particleCount,
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
    bool buildFluidBoundaryCandidatePairs(Diligent::IDeviceContext *computeContext,
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
    bool clearRigidParticleAttachmentConstraintState(Diligent::IDeviceContext *computeContext,
                                                     const PhysicsSceneGpuState &sceneState,
                                                     std::uint32_t constraintCount,
                                                     const GpuRigidDispatchConstants &constants);
    bool clearStrandRigidAttachmentConstraintState(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t constraintCount,
                                                   const GpuRigidDispatchConstants &constants);
    bool clearRigidDistanceConstraintState(Diligent::IDeviceContext *computeContext,
                                           const PhysicsSceneGpuState &sceneState,
                                           std::uint32_t constraintCount,
                                           const GpuRigidDispatchConstants &constants);
    bool clearRoutedCableConstraintState(Diligent::IDeviceContext *computeContext,
                                         const PhysicsSceneGpuState &sceneState,
                                         std::uint32_t routedCableCount,
                                         const GpuRigidDispatchConstants &constants);
    bool clearSuturingCandidates(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t suturingParticleCount,
                                 const GpuParticleDispatchConstants &constants);
    bool gatherSuturingCandidates(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  const GpuParticleDispatchConstants &constants);
    bool classifySuturingParticles(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState,
                                   std::uint32_t suturingParticleCount,
                                   const GpuParticleDispatchConstants &constants);
    bool updateSuturingTipPaths(Diligent::IDeviceContext *computeContext,
                                const PhysicsSceneGpuState &sceneState,
                                std::uint32_t suturingPairCount,
                                const GpuParticleDispatchConstants &constants);
    bool assignSuturingInsideParticles(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       std::uint32_t suturingParticleCount,
                                       const GpuParticleDispatchConstants &constants);
    bool solveSuturingNodePathConstraints(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState,
                                          std::uint32_t suturingParticleCount,
                                          const GpuParticleDispatchConstants &constants);
    bool solveSoftEdgeConstraints(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  std::uint32_t softEdgeCount,
                                  const GpuParticleDispatchConstants &constants);
    bool solveSoftBendConstraints(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  std::uint32_t softBendCount,
                                  const GpuParticleDispatchConstants &constants);
    bool solveSoftTetConstraints(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState, std::uint32_t softTetCount,
                                 const GpuParticleDispatchConstants &constants);
    bool applySoftEdgeCorrections(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  const GpuParticleDispatchConstants &constants);
    bool applySoftBendCorrections(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  const GpuParticleDispatchConstants &constants);
    bool applySoftTetCorrections(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 const GpuParticleDispatchConstants &constants);
    bool solveStrandSegmentConstraints(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       std::uint32_t strandSegmentCount,
                                       const GpuParticleDispatchConstants &constants);
    bool applyStrandSegmentCorrections(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       std::uint32_t dispatchCount,
                                       const GpuParticleDispatchConstants &constants);
    bool solveStrandJointConstraints(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t strandJointCount,
                                     const GpuParticleDispatchConstants &constants);
    bool applyStrandJointCorrections(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t dispatchCount,
                                     const GpuParticleDispatchConstants &constants);
    bool applyStrandRigidAttachmentCorrections(Diligent::IDeviceContext *computeContext,
                                               const PhysicsSceneGpuState &sceneState,
                                               std::uint32_t dispatchCount,
                                               const GpuParticleDispatchConstants &constants);
    bool solveStrandDistanceConstraints(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t strandDistanceCount,
                                        const GpuParticleDispatchConstants &constants);
    bool applyStrandDistanceCorrections(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t particleCount,
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
    bool computeFluidDensityConstraints(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        const GpuParticleDispatchConstants &constants);
    bool buildFluidNeighborPairs(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 const GpuParticleDispatchConstants &constants);
    bool computeFluidDeltaPositions(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState,
                                    const GpuParticleDispatchConstants &constants);
    bool applyFluidDeltaPositions(Diligent::IDeviceContext *computeContext,
                                  const PhysicsSceneGpuState &sceneState,
                                  const GpuParticleDispatchConstants &constants);
    bool clampFluidBoundary(Diligent::IDeviceContext *computeContext,
                            const PhysicsSceneGpuState &sceneState,
                            const GpuParticleDispatchConstants &constants);
    bool projectFluidBoundaryVelocities(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        const GpuParticleDispatchConstants &constants);
    bool computeFluidVorticity(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState,
                               const GpuParticleDispatchConstants &constants);
    bool applyFluidVorticityConfinement(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        const GpuParticleDispatchConstants &constants);
    bool buildFluidRenderAnisotropy(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState,
                                    const GpuParticleDispatchConstants &constants);
    bool solveParticleContactVelocities(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState,
                                        std::uint32_t particleCount, std::uint32_t rigidBodyCount,
                                        std::uint32_t iterations,
                                        const GpuRigidDispatchConstants &rigidConstants);
    bool solveParticleRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                             const PhysicsSceneGpuState &sceneState,
                                             std::uint32_t particleCount,
                                             std::uint32_t rigidBodyCount, std::uint32_t iterations,
                                             const GpuRigidDispatchConstants &rigidConstants);
    bool skinSoftRenderVertices(Diligent::IDeviceContext *computeContext,
                                const PhysicsSceneGpuState &sceneState,
                                std::uint32_t renderVertexCount);
    bool updateSoftTriangleNormals(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState,
                                   std::uint32_t renderTriangleCount);
    bool updateSoftRenderNormals(Diligent::IDeviceContext *computeContext,
                                 const PhysicsSceneGpuState &sceneState,
                                 std::uint32_t renderVertexCount);
    bool updateCurveRenderData(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t curveCount);
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
    bool buildRigidNarrowPhaseChunks(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState);
    bool generateRigidContacts(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState);
    bool generateProxyRigidContacts(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState,
                                    const GpuParticleDispatchConstants &constants);
    bool finalRigidContactDepenetration(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState);
    bool solveBallJointConstraints(Diligent::IDeviceContext *computeContext,
                                   const PhysicsSceneGpuState &sceneState);
    bool solveSphericalJointConstraints(Diligent::IDeviceContext *computeContext,
                                        const PhysicsSceneGpuState &sceneState);
    bool solveHingeJointConstraints(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState);
    bool solveSliderJointConstraints(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState);
    bool solveRigidParticleAttachmentConstraints(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState,
                                                 std::uint32_t constraintCount,
                                                 const GpuRigidDispatchConstants &constants);
    bool solveStrandRigidAttachmentConstraints(Diligent::IDeviceContext *computeContext,
                                               const PhysicsSceneGpuState &sceneState,
                                               std::uint32_t constraintCount,
                                               const GpuRigidDispatchConstants &constants);
    bool solveRigidDistanceConstraints(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       std::uint32_t constraintCount,
                                       const GpuRigidDispatchConstants &constants);
    bool solveRoutedCableConstraints(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t routedCableCount,
                                     const GpuRigidDispatchConstants &constants);
    bool solveHingeJointTargetVelocities(Diligent::IDeviceContext *computeContext,
                                         const PhysicsSceneGpuState &sceneState);
    bool solveSliderJointTargetVelocities(Diligent::IDeviceContext *computeContext,
                                          const PhysicsSceneGpuState &sceneState);
    bool applyRigidCorrections(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t rigidBodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool updateRigidVelocities(Diligent::IDeviceContext *computeContext,
                               const PhysicsSceneGpuState &sceneState, std::uint32_t bodyCount,
                               const GpuRigidDispatchConstants &constants);
    bool resetRigidContactVelocityAggregates(Diligent::IDeviceContext *computeContext,
                                             const PhysicsSceneGpuState &sceneState,
                                             const GpuRigidDispatchConstants &constants);
    bool initRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                    const PhysicsSceneGpuState &sceneState,
                                    const GpuRigidDispatchConstants &constants);
    bool solveRigidContactVelocities(Diligent::IDeviceContext *computeContext,
                                     const PhysicsSceneGpuState &sceneState,
                                     std::uint32_t rigidBodyCount,
                                     std::uint32_t rigidContactIterations,
                                     std::uint32_t rigidJointIterations,
                                     const GpuRigidDispatchConstants &constants);
    bool updateRigidDispatchConstants(Diligent::IDeviceContext *computeContext,
                                      const GpuRigidDispatchConstants &constants);
    bool recreateSceneBindingVariants();

private:
    bool writeRigidDispatchConstants(Diligent::IDeviceContext *computeContext,
                                     const GpuRigidDispatchConstants &constants);
    bool writeSolverConfig(Diligent::IDeviceContext *computeContext,
                           const GpuPhysicsSolverConfig &config);
    bool writeRigidJointDispatchConstants(Diligent::IDeviceContext *computeContext,
                                          const GpuRigidJointDispatchConstants &constants);
    bool writeParticleDispatchConstants(Diligent::IDeviceContext *computeContext,
                                        const GpuParticleDispatchConstants &constants);
    bool writeSoftRenderDispatchConstants(Diligent::IDeviceContext *computeContext,
                                          const GpuSoftRenderDispatchConstants &constants);
    bool writeCurveRenderDispatchConstants(Diligent::IDeviceContext *computeContext,
                                           const GpuCurveRenderDispatchConstants &constants);
    bool writeScanDispatchConstants(Diligent::IDeviceContext *computeContext,
                                    const GpuPhysicsScanDispatchConstants &constants);
    bool writeRadixConstants(Diligent::IDeviceContext *computeContext,
                             const GpuPhysicsRadixConstants &constants);
    bool writeBroadPhaseBuildConstants(Diligent::IDeviceContext *computeContext,
                                       const GpuBroadPhaseBuildConstants &constants);
    bool writeBroadPhaseReductionConstants(Diligent::IDeviceContext *computeContext,
                                           const GpuBroadPhaseReductionConstants &constants);
    bool writeConstantsBuffer(Diligent::IDeviceContext *computeContext, Diligent::IBuffer *buffer,
                              const void *constants, std::size_t constantsSize);
    bool dispatchPreparedScanBlockPass(Diligent::IDeviceContext *computeContext,
                                       Diligent::IBuffer *input, Diligent::IBuffer *output,
                                       Diligent::IBuffer *blockSums,
                                       Diligent::IBuffer *indirectArgsBuffer,
                                       std::uint32_t scanLevelIndex,
                                       std::uint32_t dispatchElementCount, bool useIndirect);
    bool dispatchPreparedScanAddOffsetsPass(Diligent::IDeviceContext *computeContext,
                                            Diligent::IBuffer *output,
                                            Diligent::IBuffer *scannedBlockOffsets,
                                            Diligent::IBuffer *indirectArgsBuffer,
                                            std::uint32_t scanLevelIndex,
                                            std::uint32_t dispatchElementCount, bool useIndirect);
    bool dispatchExclusiveScanPrepared(Diligent::IDeviceContext *computeContext,
                                       const PhysicsSceneGpuState &sceneState,
                                       Diligent::IBuffer *input, Diligent::IBuffer *output,
                                       Diligent::IBuffer *indirectArgsBuffer, bool useIndirect,
                                       const std::uint32_t *directCounts = nullptr);
    bool dispatchExclusiveScanWithCpuCount(Diligent::IDeviceContext *computeContext,
                                           const PhysicsSceneGpuState &sceneState,
                                           Diligent::IBuffer *input, Diligent::IBuffer *output,
                                           std::uint32_t count);
    bool dispatchExclusiveScanWithGpuCount(Diligent::IDeviceContext *computeContext,
                                           const PhysicsSceneGpuState &sceneState,
                                           Diligent::IBuffer *input, Diligent::IBuffer *output,
                                           bool particleRigidCandidates);
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
    bool dispatchGenerateProxyRigidContactsPass(Diligent::IDeviceContext *computeContext,
                                                const PhysicsSceneGpuState &sceneState,
                                                const GpuParticleDispatchConstants &constants);
    bool dispatchFinalRigidContactDepenetrationPass(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState);
    bool dispatchSolveBallJointConstraintsPass(Diligent::IDeviceContext *computeContext,
                                               const PhysicsSceneGpuState &sceneState,
                                               std::uint32_t jointCount);
    bool dispatchSolveSphericalJointConstraintsPass(Diligent::IDeviceContext *computeContext,
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
    bool dispatchBuildRigidNarrowPhaseChunksPass(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState);
    bool dispatchClearRigidBodyPairContactAggregatesPass(Diligent::IDeviceContext *computeContext,
                                                         const PhysicsSceneGpuState &sceneState,
                                                         std::uint32_t candidatePairCapacity);
    bool dispatchInitRigidContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                                const PhysicsSceneGpuState &sceneState);
    bool dispatchPrepareRigidContactVelocityIndirectArgsPass(
        Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState);
    bool dispatchSolveRigidContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                                 const PhysicsSceneGpuState &sceneState);
    bool dispatchApplyRigidCorrectionsPass(Diligent::IDeviceContext *computeContext,
                                           const PhysicsSceneGpuState &sceneState,
                                           std::uint32_t rigidBodyCount);
    bool dispatchSolveParticleContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                                    const PhysicsSceneGpuState &sceneState);
    bool dispatchSolveParticleRigidContactVelocitiesPass(Diligent::IDeviceContext *computeContext,
                                                         const PhysicsSceneGpuState &sceneState);
    bool dispatchClearSuturingCandidatesPass(Diligent::IDeviceContext *computeContext,
                                             const PhysicsSceneGpuState &sceneState,
                                             std::uint32_t suturingParticleCount,
                                             const GpuParticleDispatchConstants &constants);
    bool dispatchGatherSuturingCandidatesPass(Diligent::IDeviceContext *computeContext,
                                              const PhysicsSceneGpuState &sceneState,
                                              const GpuParticleDispatchConstants &constants);
    bool dispatchClassifySuturingParticlesPass(Diligent::IDeviceContext *computeContext,
                                               const PhysicsSceneGpuState &sceneState,
                                               std::uint32_t suturingParticleCount,
                                               const GpuParticleDispatchConstants &constants);
    bool dispatchUpdateSuturingTipPathsPass(Diligent::IDeviceContext *computeContext,
                                            const PhysicsSceneGpuState &sceneState,
                                            std::uint32_t suturingPairCount,
                                            const GpuParticleDispatchConstants &constants);
    bool dispatchAssignSuturingInsideParticlesPass(Diligent::IDeviceContext *computeContext,
                                                   const PhysicsSceneGpuState &sceneState,
                                                   std::uint32_t particleCount,
                                                   const GpuParticleDispatchConstants &constants);
    bool dispatchSolveSuturingNodePathConstraintsPass(
        Diligent::IDeviceContext *computeContext, const PhysicsSceneGpuState &sceneState,
        std::uint32_t particleCount, const GpuParticleDispatchConstants &constants);

    gpu::ShaderSourceProvider mShaderSourceProvider{""};
    Diligent::Uint64 mPhysicsContextMask = 0;

    gpu::GpuComputePass mRigidPredictPass;
    gpu::GpuComputePass mSoftPredictPass;
    gpu::GpuComputePass mSyncRigidProxyParticlesPass;
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
    gpu::GpuComputePass mCountFluidBoundaryCandidatePairsPass;
    gpu::GpuComputePass mFinalizeFluidBoundaryCandidatePairsPass;
    gpu::GpuComputePass mEmitFluidBoundaryCandidatePairsPass;
    gpu::GpuComputePass mGenerateParticleExplicitContactsPass;
    gpu::GpuComputePass mGenerateParticleRigidContactsPass;
    gpu::GpuComputePass mPrepareExplicitContactScanPass;
    gpu::GpuComputePass mPrepareRigidContactScanPass;
    gpu::GpuComputePass mPrepareParticleCandidateIndirectArgsPass;
    gpu::GpuComputePass mPrepareParticleActiveIndirectArgsPass;
    gpu::GpuComputePass mFinalizeActiveParticleExplicitContactsPass;
    gpu::GpuComputePass mCompactActiveParticleExplicitContactsPass;
    gpu::GpuComputePass mFinalizeActiveParticleRigidContactsPass;
    gpu::GpuComputePass mCompactActiveParticleRigidContactsPass;
    gpu::GpuComputePass mClearSoftConstraintStatePass;
    gpu::GpuComputePass mClearRigidParticleAttachmentConstraintStatePass;
    gpu::GpuComputePass mClearStrandRigidAttachmentConstraintStatePass;
    gpu::GpuComputePass mClearRigidDistanceConstraintStatePass;
    gpu::GpuComputePass mClearRoutedCableConstraintStatePass;
    gpu::GpuComputePass mClearSuturingCandidatesPass;
    gpu::GpuComputePass mGatherSuturingCandidatesPass;
    gpu::GpuComputePass mClassifySuturingParticlesPass;
    gpu::GpuComputePass mUpdateSuturingTipPathsPass;
    gpu::GpuComputePass mAssignSuturingInsideParticlesPass;
    gpu::GpuComputePass mSolveSuturingNodePathConstraintsPass;
    gpu::GpuComputePass mSolveSoftEdgeConstraintsPass;
    gpu::GpuComputePass mSolveSoftBendConstraintsPass;
    gpu::GpuComputePass mSolveSoftTetConstraintsPass;
    gpu::GpuComputePass mApplySoftEdgeCorrectionsPass;
    gpu::GpuComputePass mApplySoftBendCorrectionsPass;
    gpu::GpuComputePass mApplySoftTetCorrectionsPass;
    gpu::GpuComputePass mSolveStrandSegmentConstraintsPass;
    gpu::GpuComputePass mApplyStrandSegmentCorrectionsPass;
    gpu::GpuComputePass mSolveStrandJointConstraintsPass;
    gpu::GpuComputePass mApplyStrandJointCorrectionsPass;
    gpu::GpuComputePass mApplyStrandRigidAttachmentCorrectionsPass;
    gpu::GpuComputePass mSolveStrandDistanceConstraintsPass;
    gpu::GpuComputePass mApplyStrandDistanceCorrectionsPass;
    gpu::GpuComputePass mSolveParticleExplicitContactsPass;
    gpu::GpuComputePass mSolveParticleRigidContactsPass;
    gpu::GpuComputePass mApplyParticlePositionCorrectionsPass;
    gpu::GpuComputePass mUpdateParticleVelocitiesPass;
    gpu::GpuComputePass mBuildFluidNeighborPairsPass;
    gpu::GpuComputePass mComputeFluidDensityConstraintsPass;
    gpu::GpuComputePass mComputeFluidDeltaPositionsPass;
    gpu::GpuComputePass mApplyFluidDeltaPositionsPass;
    gpu::GpuComputePass mClampFluidBoundaryPass;
    gpu::GpuComputePass mProjectFluidBoundaryVelocitiesPass;
    gpu::GpuComputePass mComputeFluidVorticityPass;
    gpu::GpuComputePass mApplyFluidVorticityConfinementPass;
    gpu::GpuComputePass mBuildFluidRenderAnisotropyPass;
    gpu::GpuComputePass mSolveParticleContactVelocitiesPass;
    gpu::GpuComputePass mSolveParticleRigidContactVelocitiesPass;
    gpu::GpuComputePass mApplyParticleContactVelocitiesPass;
    gpu::GpuComputePass mSkinSoftRenderVerticesPass;
    gpu::GpuComputePass mUpdateSoftTriangleNormalsPass;
    gpu::GpuComputePass mUpdateSoftRenderNormalsPass;
    gpu::GpuComputePass mUpdateCurveRenderDataPass;
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
    gpu::GpuComputePass mGenerateProxyRigidContactsPass;
    gpu::GpuComputePass mClearRigidCorrectionsPass;
    gpu::GpuComputePass mFinalRigidContactDepenetrationPass;
    gpu::GpuComputePass mClearRigidBodyPairContactAggregatesPass;
    gpu::GpuComputePass mInitRigidContactVelocitiesPass;
    gpu::GpuComputePass mPrepareRigidContactVelocityIndirectArgsPass;
    gpu::GpuComputePass mSolveRigidContactVelocitiesPass;
    gpu::GpuComputePass mSolveBallJointConstraintsPass;
    gpu::GpuComputePass mSolveSphericalJointConstraintsPass;
    gpu::GpuComputePass mSolveHingeJointConstraintsPassivePass;
    gpu::GpuComputePass mSolveSliderJointConstraintsPassivePass;
    gpu::GpuComputePass mSolveRigidParticleAttachmentConstraintsPass;
    gpu::GpuComputePass mSolveStrandRigidAttachmentConstraintsPass;
    gpu::GpuComputePass mSolveRigidDistanceConstraintsPass;
    gpu::GpuComputePass mSolveRoutedCableConstraintsPass;
    gpu::GpuComputePass mSolveHingeJointTargetVelocitiesPass;
    gpu::GpuComputePass mSolveSliderJointTargetVelocitiesPass;
    gpu::GpuComputePass mUpdateHingeJointRuntimeStatePass;
    gpu::GpuComputePass mUpdateSliderJointRuntimeStatePass;
    gpu::GpuComputePass mClearHingeJointConstraintStatePass;
    gpu::GpuComputePass mClearSphericalJointConstraintStatePass;
    gpu::GpuComputePass mClearSliderJointConstraintStatePass;
    gpu::GpuComputePass mApplyRigidCorrectionsPass;
    gpu::GpuComputePass mUpdateRigidVelocitiesPass;
    gpu::GpuComputePass mApplyRigidContactVelocitiesPass;

    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRigidDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mSolverConfigBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRigidJointDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mParticleDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mSoftRenderDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mCurveRenderDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mScanDispatchConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mScanConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mScanIndirectArgsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mRadixConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseBuildConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBroadPhaseReductionConstantsBuffer;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DISPATCHER_H
