#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI

#include "physics_solver_config.hlsli"

cbuffer PhysicsParticleDispatchConstantsBuffer
{
    float dt;
    uint particleCount;
    uint rigidColliderCount;
    float particleGridCellSize;
    uint particleCandidatePairCapacity;
    uint fluidBoundaryCandidatePairCapacity;
    uint particleCellRangeCapacity;
    uint softEdgeCount;
    uint softBendCount;
    uint softTetCount;
    uint strandSegmentCount;
    uint strandJointCount;
    uint strandDistanceCount;
    uint fluidIterations;
    uint maxFluidNeighborhood;
    uint iterationIndex;
    uint suturingPairCount;
    uint suturingPathHeaderCount;
    uint suturingPathNodeCount;
    uint suturingParticleCount;
    uint maxSuturingCandidatesPerParticle;
    uint maxSuturingNodesPerPath;
    uint reserved0;
    uint reserved1;
    float4 gravity;
};

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
