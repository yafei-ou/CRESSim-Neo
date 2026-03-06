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

StructuredBuffer<uint> g_ActiveBodyFlags;
StructuredBuffer<uint> g_ActiveBodyOffsets;
RWStructuredBuffer<GpuBroadPhaseMeta> g_BroadPhaseMeta;

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuBroadPhaseMeta meta = g_BroadPhaseMeta[0];
    if (rigidBodyCount == 0u)
    {
        meta.activeDynamicCount = 0u;
    }
    else
    {
        meta.activeDynamicCount =
            g_ActiveBodyOffsets[rigidBodyCount - 1u] + g_ActiveBodyFlags[rigidBodyCount - 1u];
    }
    meta.candidatePairCount = 0u;
    meta.requiredPairCount = 0u;
    meta.overflow = 0u;
    g_BroadPhaseMeta[0] = meta;
}
