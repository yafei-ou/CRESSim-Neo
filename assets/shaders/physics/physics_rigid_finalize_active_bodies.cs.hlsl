#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ActiveBodyFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_ActiveBodyOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_StaticBodyFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_StaticBodyOffsets);
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
            CRESSIM_SB_LOAD(g_ActiveBodyOffsets, colliderCount - 1u) +
            CRESSIM_SB_LOAD(g_ActiveBodyFlags, colliderCount - 1u);
        meta.staticBodyCount =
            CRESSIM_SB_LOAD(g_StaticBodyOffsets, colliderCount - 1u) +
            CRESSIM_SB_LOAD(g_StaticBodyFlags, colliderCount - 1u);
    }
    meta.candidatePairCount = 0u;
    meta.requiredPairCount = 0u;
    meta.overflow = 0u;
    g_BroadPhaseMeta[0] = meta;
}
