#include "physics/physics_radix_constants.hlsli"
#include "physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_MortonCodesIn);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RadixBitFlags);

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= elementCount)
    {
        return;
    }

    CRESSIM_SB_STORE(g_RadixBitFlags, index, (CRESSIM_SB_LOAD(g_MortonCodesIn, index).mortonCode >> bitIndex) & 1u);
}
