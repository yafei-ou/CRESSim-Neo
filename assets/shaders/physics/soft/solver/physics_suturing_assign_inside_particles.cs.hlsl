#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint4, g_SuturingParticleRefs);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPair, g_SuturingPairs);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSuturingInsertionStateStorage, g_SuturingInsertionStates);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathHeader, g_SuturingPathHeaders);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathNode, g_SuturingPathNodes);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);

float3 EvaluatePathNodePosition(GpuSuturingPathNode node)
{
    if (node.tetIndex == kInvalidSuturingIndex || node.tetIndex >= softTetCount)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const GpuSoftTet tet = CRESSIM_SB_LOAD(g_SoftTets, node.tetIndex);
    const float3 p0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.x).xyz;
    const float3 p1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.y).xyz;
    const float3 p2 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.z).xyz;
    const float3 p3 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.w).xyz;
    return p0 * node.barycentrics.x + p1 * node.barycentrics.y + p2 * node.barycentrics.z +
           p3 * node.barycentrics.w;
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

    GpuSuturingInsertionStateStorage state = CRESSIM_SB_LOAD(g_SuturingInsertionStates, particleIndex);
    state.pathIndex = kInvalidSuturingIndex;
    state.nearestNodeIndex = kInvalidSuturingIndex;

    const uint strandRole = particleRef.y;
    if (state.state != kSuturingInsertionStateInside || strandRole == kParticleStrandRoleNone ||
        strandRole == kParticleStrandRoleNeedleTip || state.softBodyIndex == kInvalidSuturingIndex)
    {
        CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
        return;
    }

    const uint suturingGroupId = particleRef.z;
    const uint environmentIndex = particleRef.w;
    const float3 particlePosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;

    uint matchedPairIndex = kInvalidSuturingIndex;
    [loop]
    for (uint pairIndex = 0u; pairIndex < suturingPairCount; ++pairIndex)
    {
        const GpuSuturingPair pair = CRESSIM_SB_LOAD(g_SuturingPairs, pairIndex);
        if (pair.suturingGroupId == suturingGroupId && pair.environmentIndex == environmentIndex &&
            pair.softBodyIndex == state.softBodyIndex)
        {
            matchedPairIndex = pairIndex;
            break;
        }
    }

    if (matchedPairIndex == kInvalidSuturingIndex)
    {
        CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
        return;
    }

    const GpuSuturingPair pair = CRESSIM_SB_LOAD(g_SuturingPairs, matchedPairIndex);
    float bestDistanceSq = 3.402823466e+38F;
    uint bestPathIndex = kInvalidSuturingIndex;
    uint bestNodeIndex = kInvalidSuturingIndex;

    const uint headerEnd = pair.pathStart + pair.pathCount;
    [loop]
    for (uint pathIndex = pair.pathStart; pathIndex < headerEnd; ++pathIndex)
    {
        const GpuSuturingPathHeader header = CRESSIM_SB_LOAD(g_SuturingPathHeaders, pathIndex);
        if (header.suturingGroupId != pair.suturingGroupId ||
            header.softBodyIndex != pair.softBodyIndex ||
            header.nodeCount == 0u)
        {
            continue;
        }

        const uint nodeEnd = header.nodeStart + header.nodeCount;
        [loop]
        for (uint nodeIndex = header.nodeStart; nodeIndex < nodeEnd; ++nodeIndex)
        {
            const GpuSuturingPathNode node = CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex);
            const float3 nodePosition = EvaluatePathNodePosition(node);
            const float3 delta = particlePosition - nodePosition;
            const float distanceSq = dot(delta, delta);
            if (distanceSq < bestDistanceSq)
            {
                bestDistanceSq = distanceSq;
                bestPathIndex = pathIndex;
                bestNodeIndex = nodeIndex;
            }
        }
    }

    state.pathIndex = bestPathIndex;
    state.nearestNodeIndex = bestNodeIndex;
    CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
}
