#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_ActiveBodyFlags;
StructuredBuffer<uint> g_ActiveBodyOffsets;
StructuredBuffer<uint> g_StaticBodyFlags;
StructuredBuffer<uint> g_StaticBodyOffsets;
RWStructuredBuffer<GpuBroadPhaseMeta> g_BroadPhaseMeta;

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuBroadPhaseMeta meta = g_BroadPhaseMeta[0];
    if (colliderCount == 0u)
    {
        meta.activeMovingCount = 0u;
        meta.staticBodyCount = 0u;
    }
    else
    {
        meta.activeMovingCount =
            g_ActiveBodyOffsets[colliderCount - 1u] + g_ActiveBodyFlags[colliderCount - 1u];
        meta.staticBodyCount =
            g_StaticBodyOffsets[colliderCount - 1u] + g_StaticBodyFlags[colliderCount - 1u];
    }
    meta.candidatePairCount = 0u;
    meta.requiredPairCount = 0u;
    meta.overflow = 0u;
    g_BroadPhaseMeta[0] = meta;
}
