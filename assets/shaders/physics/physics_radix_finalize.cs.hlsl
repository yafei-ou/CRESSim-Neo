#include "include/physics/physics_radix_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_RadixBitFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_RadixBitOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RadixMeta);

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    if (elementCount == 0u)
    {
        CRESSIM_SB_STORE(g_RadixMeta, 0u, 0u);
        return;
    }

    const uint totalOnes = CRESSIM_SB_LOAD(g_RadixBitOffsets, elementCount - 1u) +
                           CRESSIM_SB_LOAD(g_RadixBitFlags, elementCount - 1u);
    CRESSIM_SB_STORE(g_RadixMeta, 0u, elementCount - totalOnes);
}
