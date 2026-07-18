#include "physics_rigid_broad_phase_build_constants.hlsli"
#include "physics_rigid_types.hlsli"
#include "physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_BroadPhaseBodyIndices);
CRESSIM_STRUCTURED_BUFFER(GpuBodyAabb, g_BodyAabbs);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBroadPhaseElement, g_BroadPhaseElements);

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint activeIndex = dispatchThreadID.x;
    if (activeIndex >= elementCount)
    {
        return;
    }

    const uint primitiveId = CRESSIM_SB_LOAD(g_BroadPhaseBodyIndices, activeIndex);
    const GpuBodyAabb bodyAabb = CRESSIM_SB_LOAD(g_BodyAabbs, primitiveId);

    GpuBroadPhaseElement element;
    element.primitiveIdx = primitiveId;
    element.aabbMinX = bodyAabb.minBounds.x;
    element.aabbMinY = bodyAabb.minBounds.y;
    element.aabbMinZ = bodyAabb.minBounds.z;
    element.aabbMaxX = bodyAabb.maxBounds.x;
    element.aabbMaxY = bodyAabb.maxBounds.y;
    element.aabbMaxZ = bodyAabb.maxBounds.z;
    element.reserved = 0.0;
    CRESSIM_SB_STORE(g_BroadPhaseElements, activeIndex, element);
}
