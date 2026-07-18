#include "physics_radix_constants.hlsli"
#include "physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_MortonCodesIn);
CRESSIM_STRUCTURED_BUFFER(uint, g_RadixBitFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_RadixBitOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_RadixMeta);
CRESSIM_RW_STRUCTURED_BUFFER(GpuMortonCodeElement, g_MortonCodesOut);

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
    CRESSIM_SB_STORE(g_MortonCodesOut, dstIndex, CRESSIM_SB_LOAD(g_MortonCodesIn, index));
}
