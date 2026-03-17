#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H

#include "gpu/gpu_compute_pass.h"

namespace cressim::neo::physics::passdefs
{

extern const gpu::GpuComputePassDefinition kPredict;
extern const gpu::GpuComputePassDefinition kUpdateWorldAabbs;
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
extern const gpu::GpuComputePassDefinition kGenerateContacts;
extern const gpu::GpuComputePassDefinition kSolveGather;
extern const gpu::GpuComputePassDefinition kClearCorrections;
extern const gpu::GpuComputePassDefinition kApplyCorrections;
extern const gpu::GpuComputePassDefinition kUpdateVelocities;

} // namespace cressim::neo::physics::passdefs

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_PASS_DEFINITIONS_H
