#include "include/physics/physics_rigid_dispatch_constants.hlsli"
#include "include/physics/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_BodySetFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_BodySetOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_BroadPhaseBodyIndices);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBodyMeta, g_BodyMeta);

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint primitiveIndex = dispatchThreadID.x;
    if (primitiveIndex >= colliderCount)
    {
        return;
    }

    GpuBodyMeta meta = CRESSIM_SB_LOAD(g_BodyMeta, primitiveIndex);
    if (CRESSIM_SB_LOAD(g_BodySetFlags, primitiveIndex) != 0u)
    {
        const uint activeIndex = CRESSIM_SB_LOAD(g_BodySetOffsets, primitiveIndex);
        CRESSIM_SB_STORE(g_BroadPhaseBodyIndices, activeIndex, primitiveIndex);
        meta.activeIndex = activeIndex;
    }
    else
    {
        meta.activeIndex = kInvalidIndex;
    }
    CRESSIM_SB_STORE(g_BodyMeta, primitiveIndex, meta);
}
