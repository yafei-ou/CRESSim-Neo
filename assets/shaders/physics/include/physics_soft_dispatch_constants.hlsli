#ifndef CRESSIM_NEO_PHYSICS_SOFT_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_DISPATCH_CONSTANTS_HLSLI

#include "include/structured_buffer_compat.hlsli"

cbuffer PhysicsSoftDispatchConstantsBuffer
{
    float dt;
    uint softParticleCount;
    uint rigidSurfaceParticleCount;
    float particleGridCellSize;
    uint softCandidatePairCapacity;
    uint softCellRangeCapacity;
    uint softEdgeCount;
    uint softTetCount;
};

static const float kSoftInternalRelaxation = 0.2;

#endif // CRESSIM_NEO_PHYSICS_SOFT_DISPATCH_CONSTANTS_HLSLI
