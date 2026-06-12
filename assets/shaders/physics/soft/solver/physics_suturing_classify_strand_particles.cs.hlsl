#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint4, g_SuturingParticleRefs);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwningSoftBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SuturingCandidateParticles);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleTetRanges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftIncidentTet, g_ParticleIncidentTets);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPair, g_SuturingPairs);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSuturingInsertionStateStorage, g_SuturingInsertionStates);

float4 ComputeTetBarycentrics(float3 p0, float3 p1, float3 p2, float3 p3, float3 queryPosition)
{
    const float3 v0 = p0 - p3;
    const float3 v1 = p1 - p3;
    const float3 v2 = p2 - p3;
    const float3 vp = queryPosition - p3;
    const float denom = dot(v0, cross(v1, v2));
    if (abs(denom) <= kEpsilon)
    {
        return float4(-1.0, -1.0, -1.0, -1.0);
    }

    const float b0 = dot(vp, cross(v1, v2)) / denom;
    const float b1 = dot(v0, cross(vp, v2)) / denom;
    const float b2 = dot(v0, cross(v1, vp)) / denom;
    const float b3 = 1.0 - b0 - b1 - b2;
    return float4(b0, b1, b2, b3);
}

bool BarycentricsInside(float4 bary)
{
    const float eps = 1.0e-3;
    return bary.x >= -eps && bary.y >= -eps && bary.z >= -eps && bary.w >= -eps &&
           bary.x <= 1.0 + eps && bary.y <= 1.0 + eps && bary.z <= 1.0 + eps &&
           bary.w <= 1.0 + eps;
}

bool HasMatchingSuturingPair(uint suturingGroupId, uint environmentIndex, uint softBodyIndex)
{
    [loop]
    for (uint pairIndex = 0u; pairIndex < suturingPairCount; ++pairIndex)
    {
        const GpuSuturingPair pair = CRESSIM_SB_LOAD(g_SuturingPairs, pairIndex);
        if (pair.suturingGroupId == suturingGroupId && pair.environmentIndex == environmentIndex &&
            pair.softBodyIndex == softBodyIndex)
        {
            return true;
        }
    }

    return false;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint compactIndex = dispatchThreadID.x;
    if (compactIndex >= suturingParticleCount)
    {
        return;
    }
    const uint4 particleRef = CRESSIM_SB_LOAD(g_SuturingParticleRefs, compactIndex);
    const uint particleIndex = particleRef.x;

    GpuSuturingInsertionStateStorage state;
    state.state = kSuturingInsertionStateOutside;
    state.softBodyIndex = kInvalidSuturingIndex;
    state.tetIndex = kInvalidSuturingIndex;
    state.pathIndex = kInvalidSuturingIndex;
    state.nearestNodeIndex = kInvalidSuturingIndex;
    state.reserved0 = 0u;
    state.reserved1 = 0u;
    state.reserved2 = 0u;
    state.barycentrics = float4(0.0, 0.0, 0.0, 0.0);

    const uint strandRole = particleRef.y & 0xffffu;
    if (strandRole == kParticleStrandRoleNone || suturingPairCount == 0u)
    {
        CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
        return;
    }

    const uint suturingGroupId = particleRef.z;
    const uint environmentIndex = particleRef.w;
    const float3 queryPosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;

    [loop]
    for (uint candidateIndex = 0u; candidateIndex < maxSuturingCandidatesPerParticle; ++candidateIndex)
    {
        const uint4 candidate =
            CRESSIM_SB_LOAD(g_SuturingCandidateParticles,
                            compactIndex * maxSuturingCandidatesPerParticle + candidateIndex);
        const uint otherParticleIndex = candidate.x;

        if (otherParticleIndex >= particleCount ||
            CRESSIM_SB_LOAD(g_ParticleOwnerTypes, otherParticleIndex) != kParticleOwnerTypeSoftBody)
        {
            continue;
        }

        const uint softBodyIndex = CRESSIM_SB_LOAD(g_ParticleOwningSoftBodyIndices, otherParticleIndex);
        if (softBodyIndex == kInvalidSuturingIndex ||
            !HasMatchingSuturingPair(suturingGroupId, environmentIndex, softBodyIndex))
        {
            continue;
        }

        const GpuSoftConstraintRange tetRange =
            CRESSIM_SB_LOAD(g_ParticleTetRanges, otherParticleIndex);
        [loop]
        for (uint incidentTetOffset = 0u; incidentTetOffset < tetRange.count; ++incidentTetOffset)
        {
            const GpuSoftIncidentTet incidentTet =
                CRESSIM_SB_LOAD(g_ParticleIncidentTets, tetRange.start + incidentTetOffset);
            const uint tetIndex = incidentTet.tetIndex;
            const GpuSoftTet tet = CRESSIM_SB_LOAD(g_SoftTets, tetIndex);
            const float3 p0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.x).xyz;
            const float3 p1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.y).xyz;
            const float3 p2 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.z).xyz;
            const float3 p3 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.w).xyz;
            const float4 bary = ComputeTetBarycentrics(p0, p1, p2, p3, queryPosition);
            if (!BarycentricsInside(bary))
            {
                continue;
            }

            state.state = kSuturingInsertionStateInside;
            state.softBodyIndex = softBodyIndex;
            state.tetIndex = tetIndex;
            state.barycentrics = bary;
            break;
        }

        if (state.state == kSuturingInsertionStateInside)
        {
            break;
        }
    }

    CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
}
