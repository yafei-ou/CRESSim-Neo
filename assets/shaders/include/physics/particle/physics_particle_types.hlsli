#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_TYPES_HLSLI
#define CRESSIM_NEO_PHYSICS_PARTICLE_TYPES_HLSLI

#include "../core/physics_base.hlsli"

static const uint kParticleBroadPhaseEntryTypeParticle = 0u;
static const uint kParticleKindSolid = 0u;
static const uint kParticleKindFluid = 1u;
static const uint kParticleOwnerTypeNone = 0u;
static const uint kParticleOwnerTypeSoftBody = 1u;
static const uint kParticleOwnerTypeFluidBody = 2u;
static const uint kParticleCandidatePairTypeParticleParticle = 0u;
static const uint kParticleCandidatePairTypeParticleRigid = 1u;
static const uint kParticleRigidDedupCacheSize = 16u;
static const uint kParticleRigidColliderIterationCap = 64u;
static const uint kParticlePhaseGroupMask = 0x7fffffffu;
static const uint kParticlePhaseSelfCollideFlag = 0x80000000u;

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
};

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

struct GpuSoftIncidentTet
{
    uint tetIndex;
    uint slot;
    uint reserved0;
    uint reserved1;
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

uint ParticlePhaseGroup(uint phase)
{
    return phase & kParticlePhaseGroupMask;
}

bool ParticlePhaseSelfCollideEnabled(uint phase)
{
    return (phase & kParticlePhaseSelfCollideFlag) != 0u;
}

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_TYPES_HLSLI
