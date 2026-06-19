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

float ComputeSegmentClosestDistanceSq(float3 a, float3 b, float3 queryPosition)
{
    const float3 ab = b - a;
    const float abLengthSq = dot(ab, ab);
    if (abLengthSq <= kEpsilon)
    {
        const float3 delta = queryPosition - a;
        return dot(delta, delta);
    }

    const float t = clamp(dot(queryPosition - a, ab) / abLengthSq, 0.0, 1.0);
    const float3 closestPoint = lerp(a, b, t);
    const float3 delta = queryPosition - closestPoint;
    return dot(delta, delta);
}

float ComputeClosestPointParameter(float3 a, float3 b, float3 queryPosition)
{
    const float3 ab = b - a;
    const float abLengthSq = dot(ab, ab);
    if (abLengthSq <= kEpsilon)
    {
        return 0.0;
    }

    return clamp(dot(queryPosition - a, ab) / abLengthSq, 0.0, 1.0);
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

    const uint strandRole = particleRef.y & 0xffffu;
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
        if (header.nodeCount == 1u)
        {
            const GpuSuturingPathNode node = CRESSIM_SB_LOAD(g_SuturingPathNodes, header.nodeStart);
            const float3 nodePosition = EvaluatePathNodePosition(node);
            const float3 delta = particlePosition - nodePosition;
            const float distanceSq = dot(delta, delta);
            if (distanceSq < bestDistanceSq)
            {
                bestDistanceSq = distanceSq;
                bestPathIndex = pathIndex;
                bestNodeIndex = header.nodeStart;
                state.closestSegmentTBits = asuint(0.0);
            }
            continue;
        }

        [loop]
        for (uint nodeIndex = header.nodeStart; nodeIndex + 1u < nodeEnd; ++nodeIndex)
        {
            const GpuSuturingPathNode node0 = CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex);
            const GpuSuturingPathNode node1 = CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex + 1u);
            const float3 nodePosition0 = EvaluatePathNodePosition(node0);
            const float3 nodePosition1 = EvaluatePathNodePosition(node1);
            const float distanceSq = ComputeSegmentClosestDistanceSq(
                nodePosition0, nodePosition1, particlePosition);
            if (distanceSq < bestDistanceSq)
            {
                bestDistanceSq = distanceSq;
                bestPathIndex = pathIndex;
                bestNodeIndex = nodeIndex;
                state.closestSegmentTBits =
                    asuint(ComputeClosestPointParameter(nodePosition0, nodePosition1, particlePosition));
            }
        }
    }

    state.pathIndex = bestPathIndex;
    state.nearestNodeIndex = bestNodeIndex;
    CRESSIM_SB_STORE(g_SuturingInsertionStates, particleIndex, state);
}
