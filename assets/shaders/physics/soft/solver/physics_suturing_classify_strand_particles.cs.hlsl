#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleStrandIds);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleStrandRoles);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPair, g_SuturingPairs);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandInsertionStateStorage, g_SuturingInsertionStates);

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

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    GpuStrandInsertionStateStorage state;
    state.state = kStrandInsertionStateOutside;
    state.softBodyIndex = kInvalidSuturingIndex;
    state.tetIndex = kInvalidSuturingIndex;
    state.pathIndex = kInvalidSuturingIndex;
    state.nearestNodeIndex = kInvalidSuturingIndex;
    state.reserved0 = 0u;
    state.reserved1 = 0u;
    state.reserved2 = 0u;
    state.barycentrics = float4(0.0, 0.0, 0.0, 0.0);

    const uint strandRole = CRESSIM_SB_LOAD(g_ParticleStrandRoles, particleIndex);
    if (strandRole == kParticleStrandRoleNone || suturingPairCount == 0u)
    {
        CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
        return;
    }

    const uint strandIndex = CRESSIM_SB_LOAD(g_ParticleStrandIds, particleIndex);
    const uint environmentIndex = CRESSIM_SB_LOAD(g_ParticleEnvironmentIndices, particleIndex);
    const float3 queryPosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;

    [loop]
    for (uint pairIndex = 0u; pairIndex < suturingPairCount; ++pairIndex)
    {
        const GpuSuturingPair pair = CRESSIM_SB_LOAD(g_SuturingPairs, pairIndex);
        if (pair.strandIndex != strandIndex || pair.environmentIndex != environmentIndex)
        {
            continue;
        }

        const uint tetEnd = pair.softTetStart + pair.softTetCount;
        [loop]
        for (uint tetIndex = pair.softTetStart; tetIndex < tetEnd; ++tetIndex)
        {
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

            state.state = kStrandInsertionStateInside;
            state.softBodyIndex = pair.softBodyIndex;
            state.tetIndex = tetIndex;
            state.barycentrics = bary;
            break;
        }

        if (state.state == kStrandInsertionStateInside)
        {
            break;
        }
    }

    CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
}
