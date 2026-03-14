#include "physics/include/physics_rigid_broad_phase_build_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_BroadPhaseBodyIndices;
StructuredBuffer<GpuBodyAabb> g_BodyAabbs;
RWStructuredBuffer<GpuBroadPhaseElement> g_BroadPhaseElements;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint activeIndex = dispatchThreadID.x;
    if (activeIndex >= elementCount)
    {
        return;
    }

    const uint primitiveId = g_BroadPhaseBodyIndices[activeIndex];
    const GpuBodyAabb bodyAabb = g_BodyAabbs[primitiveId];

    GpuBroadPhaseElement element;
    element.primitiveIdx = primitiveId;
    element.aabbMinX = bodyAabb.minBounds.x;
    element.aabbMinY = bodyAabb.minBounds.y;
    element.aabbMinZ = bodyAabb.minBounds.z;
    element.aabbMaxX = bodyAabb.maxBounds.x;
    element.aabbMaxY = bodyAabb.maxBounds.y;
    element.aabbMaxZ = bodyAabb.maxBounds.z;
    element.reserved = 0.0;
    g_BroadPhaseElements[activeIndex] = element;
}
