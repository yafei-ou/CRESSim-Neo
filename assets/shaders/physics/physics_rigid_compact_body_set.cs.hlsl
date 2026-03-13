#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_BodySetFlags;
StructuredBuffer<uint> g_BodySetOffsets;
RWStructuredBuffer<uint> g_BroadPhaseBodyIndices;
RWStructuredBuffer<GpuBodyMeta> g_BodyMeta;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    GpuBodyMeta meta = g_BodyMeta[bodyIndex];
    if (g_BodySetFlags[bodyIndex] != 0u)
    {
        const uint activeIndex = g_BodySetOffsets[bodyIndex];
        g_BroadPhaseBodyIndices[activeIndex] = bodyIndex;
        meta.activeIndex = activeIndex;
    }
    else
    {
        meta.activeIndex = kInvalidIndex;
    }
    g_BodyMeta[bodyIndex] = meta;
}
