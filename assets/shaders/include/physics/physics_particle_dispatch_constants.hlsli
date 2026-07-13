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
    uint cuttingToolShape;
    uint cuttingToolEnabled;
    uint cuttingToolInstantCut;
    float cuttingToolStrength;
    float3 cuttingToolTipA;
    float cuttingToolRadius;
    float3 cuttingToolTipB;
    float cuttingToolCutResistanceScale;
    float3 cuttingToolBladeCenter;
    float cuttingToolBladeHalfLength;
    float3 cuttingToolBladeAxisU;
    float cuttingToolBladeHalfDepth;
    float3 cuttingToolBladeAxisV;
    float cuttingToolBladeHalfThickness;
    float3 cuttingToolBladeNormal;
    float cuttingToolPadding;

    uint electrocauteryToolShape;
    uint electrocauteryToolEnabled;
    uint electrocauteryToolQueryReserved0;
    uint electrocauteryToolQueryReserved1;
    float3 electrocauteryToolTipA;
    float electrocauteryToolRadius;
    float3 electrocauteryToolTipB;
    float electrocauteryToolQueryReserved2;
    float3 electrocauteryToolPreviousTipA;
    float electrocauteryToolQueryReserved3;
    float3 electrocauteryToolPreviousTipB;
    float electrocauteryToolQueryReserved4;
    float3 electrocauteryToolBladeCenter;
    float electrocauteryToolBladeHalfLength;
    float3 electrocauteryToolBladeAxisU;
    float electrocauteryToolBladeHalfDepth;
    float3 electrocauteryToolBladeAxisV;
    float electrocauteryToolBladeHalfThickness;
    float3 electrocauteryToolBladeNormal;
    float electrocauteryToolQueryReserved5;
    uint electrocauteryToolMode;
    uint electrocauteryToolReserved0;
    float electrocauteryToolActiveTipLength;
    float electrocauteryToolHeatRadius;
    float electrocauteryToolAblationRadius;
    float electrocauteryToolHeatingRateCPerSecond;
    float electrocauteryToolReserved1;
    float electrocauteryToolReserved2;
};

static const float kSoftInternalRelaxation = 0.2;
static const uint kCuttingToolShapeCapsule = 0u;
static const uint kCuttingToolShapeBlade = 1u;
static const uint kElectrocauteryToolModeDisabled = 0u;
static const uint kElectrocauteryToolModeCut = 1u;
static const uint kElectrocauteryToolModeCoagulation = 2u;
static const uint kElectrocauteryToolModeBlend = 3u;

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_DISPATCH_CONSTANTS_HLSLI
