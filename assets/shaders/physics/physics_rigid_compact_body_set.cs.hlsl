#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_BodySetFlags;
StructuredBuffer<uint> g_BodySetOffsets;
RWStructuredBuffer<uint> g_BroadPhaseBodyIndices;
RWStructuredBuffer<GpuBodyMeta> g_BodyMeta;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint primitiveIndex = dispatchThreadID.x;
    if (primitiveIndex >= colliderCount)
    {
        return;
    }

    GpuBodyMeta meta = g_BodyMeta[primitiveIndex];
    if (g_BodySetFlags[primitiveIndex] != 0u)
    {
        const uint activeIndex = g_BodySetOffsets[primitiveIndex];
        g_BroadPhaseBodyIndices[activeIndex] = primitiveIndex;
        meta.activeIndex = activeIndex;
    }
    else
    {
        meta.activeIndex = kInvalidIndex;
    }
    g_BodyMeta[primitiveIndex] = meta;
}
