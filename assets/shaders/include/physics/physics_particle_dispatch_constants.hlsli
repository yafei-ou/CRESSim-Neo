#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI

cbuffer PhysicsParticleDispatchConstantsBuffer
{
    float dt;
    uint particleCount;
    uint rigidColliderCount;
    float particleGridCellSize;
    uint particleCandidatePairCapacity;
    uint particleCellRangeCapacity;
    uint softEdgeCount;
    uint softTetCount;
    uint fluidIterations;
    float reserved0;
    uint reserved1;
    uint reserved2;
};

static const float kSoftInternalRelaxation = 0.2;

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
