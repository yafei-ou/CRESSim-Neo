cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint activeMovingCount;
    uint staticBodyCount;
    uint candidatePairCount;
    uint candidatePairCapacity;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
    uint reserved0;
    uint reserved1;
};

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<GpuBroadPhaseElement> g_BroadPhaseElements;
RWStructuredBuffer<GpuBroadPhaseExtent> g_GroupExtents;

groupshared float3 g_GroupMin[64];
groupshared float3 g_GroupMax[64];

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID,
                                 uint3 groupThreadID : SV_GroupThreadID,
                                 uint3 groupID : SV_GroupID)
{
    const uint index = dispatchThreadID.x;
    const uint localIndex = groupThreadID.x;

    float3 localMin = float3(3.402823466e+38, 3.402823466e+38, 3.402823466e+38);
    float3 localMax = float3(-3.402823466e+38, -3.402823466e+38, -3.402823466e+38);
    if (index < candidatePairCapacity)
    {
        const GpuBroadPhaseElement element = g_BroadPhaseElements[index];
        localMin = float3(element.aabbMinX, element.aabbMinY, element.aabbMinZ);
        localMax = float3(element.aabbMaxX, element.aabbMaxY, element.aabbMaxZ);
    }

    g_GroupMin[localIndex] = localMin;
    g_GroupMax[localIndex] = localMax;
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 32u; offset > 0u; offset >>= 1u)
    {
        if (localIndex < offset)
        {
            g_GroupMin[localIndex] = min(g_GroupMin[localIndex], g_GroupMin[localIndex + offset]);
            g_GroupMax[localIndex] = max(g_GroupMax[localIndex], g_GroupMax[localIndex + offset]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localIndex == 0u)
    {
        GpuBroadPhaseExtent extent;
        extent.minBounds = float4(g_GroupMin[0], 0.0);
        extent.maxBounds = float4(g_GroupMax[0], 0.0);
        g_GroupExtents[groupID.x] = extent;
    }
}
