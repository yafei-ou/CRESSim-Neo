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

StructuredBuffer<GpuBroadPhaseElement> g_BroadPhaseElements;
RWStructuredBuffer<GpuMortonCodeElement> g_MortonCodes;

uint ExpandBits(uint value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

uint Morton3D(float x, float y, float z)
{
    x = clamp(x * 1024.0, 0.0, 1023.0);
    y = clamp(y * 1024.0, 0.0, 1023.0);
    z = clamp(z * 1024.0, 0.0, 1023.0);
    const uint xx = ExpandBits((uint)x);
    const uint yy = ExpandBits((uint)y);
    const uint zz = ExpandBits((uint)z);
    return xx * 4u + yy * 2u + zz;
}

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }
    if (activeDynamicCount == 0u)
    {
        return;
    }

    float3 minExtent = float3(3.402823466e+38, 3.402823466e+38, 3.402823466e+38);
    float3 maxExtent = float3(-3.402823466e+38, -3.402823466e+38, -3.402823466e+38);
    for (uint i = 0u; i < activeDynamicCount; ++i)
    {
        const GpuBroadPhaseElement element = g_BroadPhaseElements[i];
        minExtent = min(minExtent, float3(element.aabbMinX, element.aabbMinY, element.aabbMinZ));
        maxExtent = max(maxExtent, float3(element.aabbMaxX, element.aabbMaxY, element.aabbMaxZ));
    }

    const float3 extentSize =
        max(maxExtent - minExtent, float3(1.0e-5, 1.0e-5, 1.0e-5));
    for (uint i = 0u; i < activeDynamicCount; ++i)
    {
        const GpuBroadPhaseElement element = g_BroadPhaseElements[i];
        const float3 center =
            0.5 * (float3(element.aabbMinX, element.aabbMinY, element.aabbMinZ) +
                   float3(element.aabbMaxX, element.aabbMaxY, element.aabbMaxZ));
        const float3 mappedCenter = (center - minExtent) / extentSize;

        GpuMortonCodeElement morton;
        morton.mortonCode = Morton3D(mappedCenter.x, mappedCenter.y, mappedCenter.z);
        morton.elementIdx = i;
        g_MortonCodes[i] = morton;
    }
}
