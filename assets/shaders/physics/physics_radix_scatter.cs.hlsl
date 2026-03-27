#include "physics/include/physics_radix_constants.hlsli"

#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<GpuMortonCodeElement> g_MortonCodesIn;
CRESSIM_STRUCTURED_BUFFER(uint, g_RadixBitFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_RadixBitOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_RadixMeta);
RWStructuredBuffer<GpuMortonCodeElement> g_MortonCodesOut;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= elementCount)
    {
        return;
    }

    const uint flag = CRESSIM_SB_LOAD(g_RadixBitFlags, index);
    const uint offset = CRESSIM_SB_LOAD(g_RadixBitOffsets, index);
    const uint totalZeros = CRESSIM_SB_LOAD(g_RadixMeta, 0u);
    const uint dstIndex = (flag == 0u) ? (index - offset) : (totalZeros + offset);
    g_MortonCodesOut[dstIndex] = g_MortonCodesIn[index];
}
