#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H

#include "physics/physics_compute_pass.h"

namespace cressim::neo::physics::passdefs
{

extern const ComputePassDefinition kPredict;
extern const ComputePassDefinition kUpdateWorldAabbs;
extern const ComputePassDefinition kScanBlock;
extern const ComputePassDefinition kScanAddOffsets;
extern const ComputePassDefinition kCompactActiveBodies;
extern const ComputePassDefinition kFinalizeActiveBodies;
extern const ComputePassDefinition kBuildBroadPhaseElements;
extern const ComputePassDefinition kReduceExtentElements;
extern const ComputePassDefinition kReduceExtentExtents;
extern const ComputePassDefinition kMortonCodes;
extern const ComputePassDefinition kRadixClassify;
extern const ComputePassDefinition kRadixFinalize;
extern const ComputePassDefinition kRadixScatter;
extern const ComputePassDefinition kBvhHierarchy;
extern const ComputePassDefinition kBvhBoundingBoxes;
extern const ComputePassDefinition kCountPairs;
extern const ComputePassDefinition kFinalizePairs;
extern const ComputePassDefinition kEmitPairs;
extern const ComputePassDefinition kBuildNarrowPhaseChunks;
extern const ComputePassDefinition kGenerateContacts;
extern const ComputePassDefinition kSolveGather;
extern const ComputePassDefinition kClearCorrections;
extern const ComputePassDefinition kApplyCorrections;
extern const ComputePassDefinition kUpdateVelocities;

} // namespace cressim::neo::physics::passdefs

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H