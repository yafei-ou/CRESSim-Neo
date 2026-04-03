#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H

#include "gpu/gpu_compute_pass.h"

namespace cressim::neo::physics::passdefs
{

extern const gpu::GpuComputePassDefinition kPredictRigid;
extern const gpu::GpuComputePassDefinition kSoftPredict;
extern const gpu::GpuComputePassDefinition kUpdateRigidSurfaceWorldPositions;
extern const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseEntries;
extern const gpu::GpuComputePassDefinition kBuildParticleBroadPhaseKeys;
extern const gpu::GpuComputePassDefinition kMarkParticleCellRangeStarts;
extern const gpu::GpuComputePassDefinition kClearParticleCellRanges;
extern const gpu::GpuComputePassDefinition kBuildParticleCellRanges;
extern const gpu::GpuComputePassDefinition kCountSoftSoftCandidatePairs;
extern const gpu::GpuComputePassDefinition kFinalizeSoftSoftCandidatePairs;
extern const gpu::GpuComputePassDefinition kEmitSoftSoftCandidatePairs;
extern const gpu::GpuComputePassDefinition kCountSoftRigidCandidatePairs;
extern const gpu::GpuComputePassDefinition kFinalizeSoftRigidCandidatePairs;
extern const gpu::GpuComputePassDefinition kEmitSoftRigidCandidatePairs;
extern const gpu::GpuComputePassDefinition kGenerateSoftContacts;
extern const gpu::GpuComputePassDefinition kGenerateSoftRigidContacts;
extern const gpu::GpuComputePassDefinition kFinalizeActiveSoftContacts;
extern const gpu::GpuComputePassDefinition kCompactActiveSoftContacts;
extern const gpu::GpuComputePassDefinition kFinalizeActiveSoftRigidContacts;
extern const gpu::GpuComputePassDefinition kCompactActiveSoftRigidContacts;
extern const gpu::GpuComputePassDefinition kClearSoftConstraintState;
extern const gpu::GpuComputePassDefinition kSolveSoftEdgeConstraints;
extern const gpu::GpuComputePassDefinition kSolveSoftTetConstraints;
extern const gpu::GpuComputePassDefinition kSolveSoftContacts;
extern const gpu::GpuComputePassDefinition kSolveSoftRigidContacts;
extern const gpu::GpuComputePassDefinition kApplySoftPositionCorrections;
extern const gpu::GpuComputePassDefinition kUpdateSoftVelocities;
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
extern const gpu::GpuComputePassDefinition kGenerateRigidContacts;
extern const gpu::GpuComputePassDefinition kSolveRigidContactConstraints;
extern const gpu::GpuComputePassDefinition kClearRigidCorrections;
extern const gpu::GpuComputePassDefinition kApplyRigidCorrections;
extern const gpu::GpuComputePassDefinition kUpdateRigidVelocities;
extern const gpu::GpuComputePassDefinition kSolveRigidContactVelocities;
extern const gpu::GpuComputePassDefinition kApplyRigidContactVelocities;

} // namespace cressim::neo::physics::passdefs

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H
