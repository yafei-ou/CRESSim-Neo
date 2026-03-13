#include "physics/include/physics_radix_constants.hlsli"

StructuredBuffer<uint> g_RadixBitFlags;
StructuredBuffer<uint> g_RadixBitOffsets;
RWStructuredBuffer<uint> g_RadixMeta;

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    if (elementCount == 0u)
    {
        g_RadixMeta[0] = 0u;
        return;
    }

    const uint totalOnes = g_RadixBitOffsets[elementCount - 1u] + g_RadixBitFlags[elementCount - 1u];
    g_RadixMeta[0] = elementCount - totalOnes;
}
