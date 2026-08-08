#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_TYPES_HLSLI
#define CRESSIM_NEO_PHYSICS_PARTICLE_TYPES_HLSLI

#include "physics_base.hlsli"

static const uint kParticleBroadPhaseEntryTypeParticle = 0u;
static const uint kParticleKindSolid = 0u;
static const uint kParticleKindFluid = 1u;
static const uint kParticleOwnerTypeNone = 0u;
static const uint kParticleOwnerTypeSoftBody = 1u;
static const uint kParticleOwnerTypeFluidBody = 2u;
static const uint kParticleOwnerTypeStrand = 3u;
static const uint kParticleOwnerTypeRigidBody = 4u;
static const uint kParticleStrandRoleNone = 0u;
static const uint kParticleStrandRoleNeedleTip = 1u;
static const uint kParticleStrandRoleNeedleBody = 2u;
static const uint kParticleStrandRoleThread = 3u;
static const uint kSuturingInsertionStateOutside = 0u;
static const uint kSuturingInsertionStateInside = 1u;
static const uint kInvalidSuturingIndex = 0xffffffffu;
static const uint kParticleCandidatePairTypeParticleParticle = 0u;
static const uint kParticleCandidatePairTypeParticleRigid = 1u;
static const uint kParticleRigidDedupCacheSize = 16u;
static const uint kParticleRigidColliderIterationCap = 64u;
static const uint kParticlePhaseGroupMask = 0x7fffffffu;
static const uint kParticlePhaseSelfCollideFlag = 0x80000000u;
static const uint kShapeClusterActive = 1u << 0u;
static const uint kShapeClusterCutDisabled = 1u << 1u;
static const uint kShapeClusterDegenerate = 1u << 2u;
static const uint kSoftEdgeActiveFlag = 1u << 0u;
static const uint kSoftEdgeCutFlag = 1u << 1u;
static const uint kSoftEdgeFracturedFlag = 1u << 2u;
static const uint kSoftEdgeDisabledFlag = 1u << 3u;
static const uint kSoftEdgeThermalCutFlag = 1u << 4u;
static const uint kSoftEdgeThermallySeverableFlag = 1u << 5u;

struct GpuParticleBroadPhaseEntry
{
    uint cellKey;
    int cellX;
    int cellY;
    int cellZ;
    uint particleIndex;
    uint particleType;
    uint reserved0;
    uint reserved1;
};

struct GpuParticleCandidatePair
{
    uint pairType;
    uint indexA;
    uint indexB;
    uint auxIndex;
};

struct GpuParticleNeighborMeta
{
    uint particleParticleCandidateCount;
    uint particleRigidCandidateCount;
    uint fluidBoundaryCandidateCount;
    uint requiredParticleParticleCandidateCount;
    uint requiredParticleRigidCandidateCount;
    uint requiredFluidBoundaryCandidateCount;
    uint particleParticleCandidateOverflow;
    uint particleRigidCandidateOverflow;
    uint fluidBoundaryCandidateOverflow;
    uint activeParticleContactCount;
    uint activeParticleRigidContactCount;
    uint reserved0;
    uint reserved1;
};

struct GpuParticleCellRange
{
    uint cellKey;
    uint startIndex;
    uint endIndex;
    uint reserved0;
};

struct GpuParticleRigidContact
{
    uint particleIndex;
    uint rigidBodyIndex;
    uint colliderIndex;
    uint active;
    float4 normalPenetration;
    float4 rigidLocalPoint;
    float4 material;
};

struct GpuParticleContact
{
    uint particleA;
    uint particleB;
    uint active;
    uint reserved0;
    float4 normalPenetration;
    float4 material;
};

struct GpuFluidMaterial
{
    float restDensity;
    float invRestDensity;
    float smoothingRadius;
    float densityConstraintScaleDerived;
    float viscosityDerived;
    float cohesionDerived;
    float cohesion1;
    float cohesion2;
    float surfaceTensionDerived;
    float vorticityConfinementDerived;
    float gravityScale;
    float cflRadius;
};

struct GpuSoftEdge
{
    uint particleA;
    uint particleB;
    float restLength;
    float compliance;
    float damage;
    float strain;
    float failureThreshold;
    float cutResistance;
    uint flags;
    float referenceRestLength;
    float referenceFailureThreshold;
    float referenceCutResistance;
    float referenceCompliance;
};

struct GpuSoftParticleThermalState
{
    float temperatureC;
    float arrheniusOmega;
    float thermalDamage;
    float waterFraction;
    float maximumTemperatureC;
    float charLevel;
    float reserved0;
    float reserved1;
};

struct GpuSoftThermalMaterial
{
    float bodyTemperatureC;
    float maximumTemperatureC;
    float diffusionRate;
    float coolingRate;

    float metersPerWorldUnit;
    float densityKgPerM3;
    float specificHeatJPerKgK;
    float thermalConductivityWPerMK;

    float bloodDensityKgPerM3;
    float bloodSpecificHeatJPerKgK;
    float bloodPerfusionPerSecond;
    float bloodTemperatureC;

    float metabolicHeatWPerM3;
    float logArrheniusA;
    float activationEnergyJPerMol;
    float coagulationOmegaStart;

    float irreversibleDamageOmega;
    float thermalCutOmega;
    float reserved0;
    float reserved1;

    float damageStartTemperatureC;
    float damageFullTemperatureC;
    float damageRate;
    uint damageModel;

    float evaporationStartTemperatureC;
    float evaporationTransitionWidthC;
    float evaporationRate;
    float reserved3;

    float charStartTemperatureC;
    float charFullTemperatureC;
    float charRate;
    float reserved4;

    float maximumShrinkage;
    float shrinkageRate;
    float shrinkDamageStart;
    float shrinkDamageFull;

    float minimumFailureThresholdScale;
    float minimumCutResistanceScale;
    float thermalCutDamageThreshold;
    float thermalCutWaterThreshold;

    float maximumComplianceMultiplier;
    float reserved5;
    float reserved6;
    float reserved7;
};

static const uint kThermalDamageModelThresholdRate = 0u;
static const uint kThermalDamageModelArrhenius = 1u;

struct GpuSoftTet
{
    uint4 particleIndices;
    float restVolume;
    float compliance;
    uint reserved0;
    uint reserved1;
};

struct GpuSoftBend
{
    uint particle0;
    uint particle1;
    uint particle2;
    float restAngle;
    float compliance;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct GpuShapeCluster
{
    uint memberOffset;
    uint memberCount;
    uint linkOffset;
    uint linkCount;
    float4 restCenterAndMass;
    uint flags;
    float stiffness;
    float compliance;
    uint padding;
};

struct GpuShapeClusterMember
{
    uint particleIndex;
    float fittingWeight;
    float blendWeight;
    uint padding0;
    float4 restOffset;
};

struct GpuShapeClusterLink
{
    uint softEdgeIndex;
    uint localParticleA;
    uint localParticleB;
    uint padding;
};

struct GpuParticleShapeMembershipRange
{
    uint offset;
    uint count;
};

struct GpuShapeClusterPose
{
    float4 rotationQuaternion;
    float4 currentCenterAndStatus;
};

struct GpuSoftConstraintRange
{
    uint start;
    uint count;
    uint reserved0;
    uint reserved1;
};

struct GpuSoftIncidentEdge
{
    uint edgeIndex;
    uint slot;
    uint reserved0;
    uint reserved1;
};

struct GpuSoftThermalAdjacency
{
    uint neighbourParticleIndex;
    uint connectingEdgeIndex;
    uint edgeFlags;
    uint reserved0;
};

uint GetSoftEdgeOtherParticle(GpuSoftEdge edge, uint particleIndex)
{
    return edge.particleA == particleIndex ? edge.particleB : edge.particleA;
}

bool IsSoftEdgeTopologyConductive(GpuSoftEdge edge)
{
    return (edge.flags & kSoftEdgeActiveFlag) != 0u &&
           (edge.flags & kSoftEdgeDisabledFlag) == 0u &&
           (edge.flags & kSoftEdgeCutFlag) == 0u &&
           (edge.flags & kSoftEdgeThermalCutFlag) == 0u &&
           (edge.flags & kSoftEdgeFracturedFlag) == 0u;
}

GpuSoftThermalAdjacency MakeSoftThermalAdjacency(uint particleIndex,
                                                 GpuSoftIncidentEdge incidentEdge,
                                                 GpuSoftEdge edge)
{
    GpuSoftThermalAdjacency adjacency;
    adjacency.neighbourParticleIndex = GetSoftEdgeOtherParticle(edge, particleIndex);
    adjacency.connectingEdgeIndex = incidentEdge.edgeIndex;
    adjacency.edgeFlags = edge.flags;
    adjacency.reserved0 = 0u;
    return adjacency;
}

struct GpuSoftIncidentTet
{
    uint tetIndex;
    uint slot;
    uint reserved0;
    uint reserved1;
};

struct GpuStrandIncidentSegment
{
    uint segmentIndex;
    uint slot;
    uint reserved0;
    uint reserved1;
};

struct GpuStrandIncidentJoint
{
    uint jointIndex;
    uint slot;
    uint reserved0;
    uint reserved1;
};

struct GpuStrandIncidentAttachment
{
    uint attachmentIndex;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct GpuSoftIncidentBend
{
    uint bendIndex;
    uint slot;
    uint reserved0;
    uint reserved1;
};

struct GpuSoftEdgeCorrection
{
    float4 correctionA;
    float4 correctionB;
};

struct GpuSoftTetCorrection
{
    float4 correction0;
    float4 correction1;
    float4 correction2;
    float4 correction3;
};

struct GpuSoftBendCorrection
{
    float4 correction0;
    float4 correction1;
    float4 correction2;
};

struct GpuStrandSegment
{
    uint particleA;
    uint particleB;
    float restLength;
    float stretchShearCompliance;
    float4 restOrientation;
};

struct GpuStrandJoint
{
    uint segmentA;
    uint segmentB;
    float bendCompliance;
    float twistCompliance;
    float4 restRelativeOrientation;
};

struct GpuStrandSegmentState
{
    float4 orientation;
};

struct GpuStrandSegmentCorrection
{
    float4 correctionA;
    float4 correctionB;
    float4 angularCorrection;
};

struct GpuStrandJointCorrection
{
    float4 correction0;
    float4 correction1;
    float4 correction2;
    float4 twistRotationA;
    float4 twistRotationB;
};

struct GpuStrandRigidAttachmentLambda
{
    float4 translation;
    float4 rotation;
};

struct GpuStrandRigidAttachmentCorrection
{
    float4 segmentRotation;
};

struct GpuStrandDistanceConstraint
{
    uint particleA;
    uint particleB;
    float restLength;
    float distanceCompliance;
};

struct GpuSoftBodyParticleRange
{
    uint start;
    uint count;
    uint reserved0;
    uint reserved1;
};

struct GpuSoftBodyChunkRange
{
    uint start;
    uint count;
    uint reserved0;
    uint reserved1;
};

struct GpuSoftBodyBoundsChunk
{
    uint softBodyIndex;
    uint particleStart;
    uint particleCount;
    uint reserved0;
};

struct GpuSuturingPair
{
    uint suturingGroupId;
    uint softBodyIndex;
    uint strandParticleStart;
    uint strandParticleCount;
    uint tipParticleIndex;
    uint softTetStart;
    uint softTetCount;
    uint pathStart;
    uint pathCount;
    uint nodeStart;
    uint nodeCount;
    uint activePathIndex;
    uint environmentIndex;
    float pathNodeSpacing;
    uint reserved0;
    uint reserved1;
};

struct GpuSuturingPathHeader
{
    uint suturingGroupId;
    uint softBodyIndex;
    uint nodeStart;
    uint nodeCount;
    uint flags;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct GpuSuturingPathNode
{
    uint softBodyIndex;
    uint tetIndex;
    float4 barycentrics;
    float4 tangentArcLength;
    uint reserved0;
    uint reserved1;
};

struct GpuSuturingInsertionStateStorage
{
    uint state;
    uint softBodyIndex;
    uint tetIndex;
    uint pathIndex;
    uint nearestNodeIndex;
    uint closestSegmentTBits;
    uint reserved1;
    uint reserved2;
    float4 barycentrics;
};

uint ParticlePhaseGroup(uint phase)
{
    return phase & kParticlePhaseGroupMask;
}

bool ParticlePhaseSelfCollideEnabled(uint phase)
{
    return (phase & kParticlePhaseSelfCollideFlag) != 0u;
}

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_TYPES_HLSLI
