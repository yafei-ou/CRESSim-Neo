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

StructuredBuffer<GpuBroadPhaseExtent> g_InputExtents;
RWStructuredBuffer<GpuBroadPhaseExtent> g_OutputExtents;

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
        const GpuBroadPhaseExtent extent = g_InputExtents[index];
        localMin = extent.minBounds.xyz;
        localMax = extent.maxBounds.xyz;
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
        g_OutputExtents[groupID.x] = extent;
    }
}
