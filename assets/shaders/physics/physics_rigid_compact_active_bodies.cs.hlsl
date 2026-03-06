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
RWStructuredBuffer<uint> g_ActiveBodyIndices;
RWStructuredBuffer<GpuBodyMeta> g_BodyMeta;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    GpuBodyMeta meta = g_BodyMeta[bodyIndex];
    if (g_ActiveBodyFlags[bodyIndex] != 0u)
    {
        const uint activeIndex = g_ActiveBodyOffsets[bodyIndex];
        g_ActiveBodyIndices[activeIndex] = bodyIndex;
        meta.activeIndex = activeIndex;
    }
    else
    {
        meta.activeIndex = kInvalidIndex;
    }
    g_BodyMeta[bodyIndex] = meta;
}
