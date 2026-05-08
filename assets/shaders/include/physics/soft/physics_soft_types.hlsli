#ifndef CRESSIM_NEO_PHYSICS_SOFT_TYPES_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_TYPES_HLSLI

#include "../core/physics_base.hlsli"

static const uint kParticleBroadPhaseEntryTypeSoft = 0u;
static const uint kSoftCandidatePairTypeSoftSoft = 0u;
static const uint kSoftCandidatePairTypeSoftRigid = 1u;
static const uint kSoftRigidDedupCacheSize = 16u;
static const uint kSoftRigidColliderIterationCap = 64u;
static const uint kSoftPhaseGroupMask = 0x7fffffffu;
static const uint kSoftPhaseSelfCollideFlag = 0x80000000u;

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

struct GpuSoftCandidatePair
{
    uint pairType;
    uint indexA;
    uint indexB;
    uint auxIndex;
};

struct GpuSoftNeighborMeta
{
    uint softSoftCandidateCount;
    uint softRigidCandidateCount;
    uint requiredSoftSoftCandidateCount;
    uint requiredSoftRigidCandidateCount;
    uint softSoftCandidateOverflow;
    uint softRigidCandidateOverflow;
    uint activeSoftContactCount;
    uint activeSoftRigidContactCount;
};

struct GpuParticleCellRange
{
    uint cellKey;
    uint startIndex;
    uint endIndex;
    uint reserved0;
};

struct GpuSoftRigidContact
{
    uint softParticleIndex;
    uint rigidBodyIndex;
    uint colliderIndex;
    uint active;
    float4 normalPenetration;
    float4 rigidLocalPoint;
    float4 material;
};

struct GpuSoftContact
{
    uint particleA;
    uint particleB;
    uint active;
    uint reserved0;
    float4 normalPenetration;
    float4 material;
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

uint SoftParticlePhaseGroup(uint phase)
{
    return phase & kSoftPhaseGroupMask;
}

bool SoftParticlePhaseSelfCollideEnabled(uint phase)
{
    return (phase & kSoftPhaseSelfCollideFlag) != 0u;
}

#endif // CRESSIM_NEO_PHYSICS_SOFT_TYPES_HLSLI
