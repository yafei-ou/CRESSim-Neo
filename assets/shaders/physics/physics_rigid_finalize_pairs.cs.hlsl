cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint activeDynamicCount;
    uint candidatePairCount;
    uint candidatePairCapacity;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
};

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_PairCounts;
StructuredBuffer<uint> g_PairOffsets;
RWStructuredBuffer<GpuBroadPhaseMeta> g_BroadPhaseMeta;

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuBroadPhaseMeta meta = g_BroadPhaseMeta[0];
    if (activeDynamicCount == 0u)
    {
        meta.candidatePairCount = 0u;
        meta.requiredPairCount = 0u;
        meta.overflow = 0u;
    }
    else
    {
        const uint totalPairs =
            g_PairOffsets[activeDynamicCount - 1u] + g_PairCounts[activeDynamicCount - 1u];
        meta.candidatePairCount = totalPairs;
        meta.requiredPairCount = totalPairs;
        meta.overflow = totalPairs > candidatePairCapacity ? 1u : 0u;
    }
    g_BroadPhaseMeta[0] = meta;
}
