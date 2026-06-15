#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI

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
};

static const float kSoftInternalRelaxation = 0.2;

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
