#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingInsertionStateStorage, g_SuturingInsertionStates);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSuturingPair, g_SuturingPairs);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSuturingPathHeader, g_SuturingPathHeaders);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSuturingPathNode, g_SuturingPathNodes);

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
    const uint pairIndex = dispatchThreadID.x;
    if (pairIndex >= suturingPairCount)
    {
        return;
    }

    GpuSuturingPair pair = CRESSIM_SB_LOAD(g_SuturingPairs, pairIndex);
    if (pair.tipParticleIndex >= particleCount || pair.pathCount == 0u)
    {
        return;
    }

    const GpuSuturingInsertionStateStorage tipState =
        CRESSIM_SB_LOAD(g_SuturingInsertionStates, pair.tipParticleIndex);
    if (tipState.state != kSuturingInsertionStateInside || tipState.softBodyIndex != pair.softBodyIndex)
    {
        pair.activePathIndex = kInvalidSuturingIndex;
        CRESSIM_SB_STORE(g_SuturingPairs, pairIndex, pair);
        return;
    }

    const uint nodesPerPath = max(pair.nodeCount / max(pair.pathCount, 1u), 1u);
    uint activePathIndex = pair.activePathIndex;
    if (activePathIndex == kInvalidSuturingIndex)
    {
        [loop]
        for (uint pathIndex = pair.pathStart; pathIndex < pair.pathStart + pair.pathCount; ++pathIndex)
        {
            GpuSuturingPathHeader header = CRESSIM_SB_LOAD(g_SuturingPathHeaders, pathIndex);
            if (header.suturingGroupId != kInvalidSuturingIndex)
            {
                continue;
            }

            const uint localPath = pathIndex - pair.pathStart;
            header.suturingGroupId = pair.suturingGroupId;
            header.softBodyIndex = pair.softBodyIndex;
            header.nodeStart = pair.nodeStart + localPath * nodesPerPath;
            header.nodeCount = 1u;
            header.flags = 1u;
            header.reserved0 = pair.reserved0;
            CRESSIM_SB_STORE(g_SuturingPathHeaders, pathIndex, header);

            GpuSuturingPathNode node;
            node.softBodyIndex = pair.softBodyIndex;
            node.tetIndex = tipState.tetIndex;
            node.barycentrics = tipState.barycentrics;
            node.tangentArcLength = float4(0.0, 0.0, 1.0, 0.0);
            CRESSIM_SB_STORE(g_SuturingPathNodes, header.nodeStart, node);

            activePathIndex = pathIndex;
            pair.activePathIndex = pathIndex;
            CRESSIM_SB_STORE(g_SuturingPairs, pairIndex, pair);
            return;
        }

        return;
    }

    GpuSuturingPathHeader header = CRESSIM_SB_LOAD(g_SuturingPathHeaders, activePathIndex);
    if (header.nodeCount == 0u)
    {
        return;
    }

    const uint lastNodeIndex = header.nodeStart + header.nodeCount - 1u;
    const GpuSuturingPathNode lastNode = CRESSIM_SB_LOAD(g_SuturingPathNodes, lastNodeIndex);
    const float3 tipPosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, pair.tipParticleIndex).xyz;
    const float3 lastPosition = EvaluatePathNodePosition(lastNode);
    const float3 delta = tipPosition - lastPosition;
    const float distance = length(delta);
    if (distance < max(pair.pathNodeSpacing, 1.0e-4))
    {
        return;
    }
    if (header.nodeCount >= nodesPerPath)
    {
        return;
    }

    GpuSuturingPathNode node;
    node.softBodyIndex = pair.softBodyIndex;
    node.tetIndex = tipState.tetIndex;
    node.barycentrics = tipState.barycentrics;
    const float3 tangent = distance > kEpsilon ? normalize(delta) : float3(0.0, 0.0, 1.0);
    node.tangentArcLength = float4(tangent, lastNode.tangentArcLength.w + distance);

    CRESSIM_SB_STORE(g_SuturingPathNodes, header.nodeStart + header.nodeCount, node);
    header.nodeCount += 1u;
    CRESSIM_SB_STORE(g_SuturingPathHeaders, activePathIndex, header);
}
