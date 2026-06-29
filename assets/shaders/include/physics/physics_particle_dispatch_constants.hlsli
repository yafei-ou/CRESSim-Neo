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
    uint shapeClusterCount;
    uint shapeRotationIterations;
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
    float maximumShapeCorrection;
    uint particleShapeMembershipIndexCount;
    uint reserved0;
    uint reserved1;
    float4 gravity;
    float3 cuttingToolTipA;
    float cuttingToolRadius;
    float3 cuttingToolTipB;
    float cuttingToolStrength;
    float cuttingToolCutThreshold;
    float cuttingToolPadding;
    uint cuttingToolEnabled;
    uint cuttingToolReserved0;
};

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
