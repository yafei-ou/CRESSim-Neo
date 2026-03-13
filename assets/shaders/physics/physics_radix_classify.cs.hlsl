#include "physics/include/physics_radix_constants.hlsli"

#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<GpuMortonCodeElement> g_MortonCodesIn;
RWStructuredBuffer<uint> g_RadixBitFlags;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= elementCount)
    {
        return;
    }

    g_RadixBitFlags[index] = (g_MortonCodesIn[index].mortonCode >> bitIndex) & 1u;
}
