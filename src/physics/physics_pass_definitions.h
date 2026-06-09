#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H

#include "gpu/gpu_compute_pass.h"

namespace cressim::neo::physics::passdefs
{

extern const gpu::GpuComputePassDefinition kPredictRigid;
extern const gpu::GpuComputePassDefinition kSoftPredict;
extern const gpu::GpuComputePassDefinition kSyncRigidProxyParticles;
extern const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseEntries;
extern const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseKeys;
extern const gpu::GpuComputePassDefinition kMarkParticleCellRangeStarts;
extern const gpu::GpuComputePassDefinition kClearParticleCellRanges;
extern const gpu::GpuComputePassDefinition kBuildParticleCellRanges;
extern const gpu::GpuComputePassDefinition kCountParticleParticleCandidatePairs;
extern const gpu::GpuComputePassDefinition kFinalizeParticleParticleCandidatePairs;
extern const gpu::GpuComputePassDefinition kEmitParticleParticleCandidatePairs;
extern const gpu::GpuComputePassDefinition kCountParticleRigidCandidatePairs;
extern const gpu::GpuComputePassDefinition kFinalizeParticleRigidCandidatePairs;
extern const gpu::GpuComputePassDefinition kEmitParticleRigidCandidatePairs;
extern const gpu::GpuComputePassDefinition kCountFluidBoundaryCandidatePairs;
extern const gpu::GpuComputePassDefinition kFinalizeFluidBoundaryCandidatePairs;
extern const gpu::GpuComputePassDefinition kEmitFluidBoundaryCandidatePairs;
extern const gpu::GpuComputePassDefinition kGenerateParticleExplicitContacts;
extern const gpu::GpuComputePassDefinition kGenerateParticleRigidContacts;
extern const gpu::GpuComputePassDefinition kPrepareExplicitContactScan;
extern const gpu::GpuComputePassDefinition kPrepareRigidContactScan;
extern const gpu::GpuComputePassDefinition kPrepareParticleCandidateIndirectArgs;
extern const gpu::GpuComputePassDefinition kPrepareParticleActiveIndirectArgs;
extern const gpu::GpuComputePassDefinition kFinalizeActiveParticleExplicitContacts;
extern const gpu::GpuComputePassDefinition kCompactActiveParticleExplicitContacts;
extern const gpu::GpuComputePassDefinition kFinalizeActiveParticleRigidContacts;
extern const gpu::GpuComputePassDefinition kCompactActiveParticleRigidContacts;
extern const gpu::GpuComputePassDefinition kClearSoftConstraintState;
extern const gpu::GpuComputePassDefinition kClearSuturingCandidates;
extern const gpu::GpuComputePassDefinition kGatherSuturingCandidates;
extern const gpu::GpuComputePassDefinition kClassifySuturingParticles;
extern const gpu::GpuComputePassDefinition kUpdateSuturingTipPaths;
extern const gpu::GpuComputePassDefinition kAssignSuturingInsideParticles;
extern const gpu::GpuComputePassDefinition kSolveSuturingNodePathConstraints;
extern const gpu::GpuComputePassDefinition kSolveSoftEdgeConstraints;
extern const gpu::GpuComputePassDefinition kSolveSoftBendConstraints;
extern const gpu::GpuComputePassDefinition kSolveSoftTetConstraints;
extern const gpu::GpuComputePassDefinition kApplySoftEdgeCorrections;
extern const gpu::GpuComputePassDefinition kApplySoftBendCorrections;
extern const gpu::GpuComputePassDefinition kApplySoftTetCorrections;
extern const gpu::GpuComputePassDefinition kSolveParticleExplicitContacts;
extern const gpu::GpuComputePassDefinition kSolveParticleRigidContacts;
extern const gpu::GpuComputePassDefinition kApplyParticlePositionCorrections;
extern const gpu::GpuComputePassDefinition kUpdateParticleVelocities;
extern const gpu::GpuComputePassDefinition kBuildFluidNeighborPairs;
extern const gpu::GpuComputePassDefinition kComputeFluidDensityConstraints;
extern const gpu::GpuComputePassDefinition kComputeFluidDeltaPositions;
extern const gpu::GpuComputePassDefinition kApplyFluidDeltaPositions;
extern const gpu::GpuComputePassDefinition kClampFluidBoundary;
extern const gpu::GpuComputePassDefinition kProjectFluidBoundaryVelocities;
extern const gpu::GpuComputePassDefinition kComputeFluidVorticity;
extern const gpu::GpuComputePassDefinition kApplyFluidVorticityConfinement;
extern const gpu::GpuComputePassDefinition kBuildFluidRenderAnisotropy;
extern const gpu::GpuComputePassDefinition kSolveParticleContactVelocities;
extern const gpu::GpuComputePassDefinition kSolveParticleRigidContactVelocities;
extern const gpu::GpuComputePassDefinition kApplyParticleContactVelocities;
extern const gpu::GpuComputePassDefinition kUpdateSoftTriangleNormals;
extern const gpu::GpuComputePassDefinition kUpdateSoftRenderNormals;
extern const gpu::GpuComputePassDefinition kUpdateSoftBodyBounds;
extern const gpu::GpuComputePassDefinition kFinalizeSoftBodyBounds;
extern const gpu::GpuComputePassDefinition kUpdateRigidWorldAabbs;
extern const gpu::GpuComputePassDefinition kScanBlock;
extern const gpu::GpuComputePassDefinition kScanAddOffsets;
extern const gpu::GpuComputePassDefinition kCompactBodySet;
extern const gpu::GpuComputePassDefinition kFinalizeActiveBodies;
extern const gpu::GpuComputePassDefinition kBuildBroadPhaseElements;
extern const gpu::GpuComputePassDefinition kReduceExtentElements;
extern const gpu::GpuComputePassDefinition kReduceExtentExtents;
extern const gpu::GpuComputePassDefinition kMortonCodes;
extern const gpu::GpuComputePassDefinition kRadixClassify;
extern const gpu::GpuComputePassDefinition kRadixFinalize;
extern const gpu::GpuComputePassDefinition kRadixScatter;
extern const gpu::GpuComputePassDefinition kBvhHierarchy;
extern const gpu::GpuComputePassDefinition kBvhBoundingBoxes;
extern const gpu::GpuComputePassDefinition kCountPairs;
extern const gpu::GpuComputePassDefinition kFinalizePairs;
extern const gpu::GpuComputePassDefinition kEmitPairs;
extern const gpu::GpuComputePassDefinition kBuildNarrowPhaseChunks;
extern const gpu::GpuComputePassDefinition kPrepareRigidIndirectArgs;
extern const gpu::GpuComputePassDefinition kGenerateRigidContacts;
extern const gpu::GpuComputePassDefinition kGenerateProxyRigidContacts;
extern const gpu::GpuComputePassDefinition kFinalRigidContactDepenetration;
extern const gpu::GpuComputePassDefinition kClearRigidBodyPairContactAggregates;
extern const gpu::GpuComputePassDefinition kInitRigidContactVelocities;
extern const gpu::GpuComputePassDefinition kPrepareRigidContactVelocityIndirectArgs;
extern const gpu::GpuComputePassDefinition kSolveRigidContactVelocities;
extern const gpu::GpuComputePassDefinition kSolveBallJointConstraints;
extern const gpu::GpuComputePassDefinition kSolveHingeJointConstraintsPassive;
extern const gpu::GpuComputePassDefinition kSolveHingeJointConstraintsTargetPosition;
extern const gpu::GpuComputePassDefinition kSolveSliderJointConstraintsPassive;
extern const gpu::GpuComputePassDefinition kSolveSliderJointConstraintsTargetPosition;
extern const gpu::GpuComputePassDefinition kSolveHingeJointTargetVelocities;
extern const gpu::GpuComputePassDefinition kSolveSliderJointTargetVelocities;
extern const gpu::GpuComputePassDefinition kClearHingeJointConstraintState;
extern const gpu::GpuComputePassDefinition kClearSliderJointConstraintState;
extern const gpu::GpuComputePassDefinition kClearRigidCorrections;
extern const gpu::GpuComputePassDefinition kApplyRigidCorrections;
extern const gpu::GpuComputePassDefinition kUpdateRigidVelocities;
extern const gpu::GpuComputePassDefinition kApplyRigidContactVelocities;

} // namespace cressim::neo::physics::passdefs

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H
