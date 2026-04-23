#include "../../../include/physics/soft/physics_soft_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuSoftContact, g_SoftContacts);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveOffsets);
CRESSIM_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftContact, g_ActiveSoftContacts);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    const GpuSoftNeighborMeta meta = CRESSIM_SB_LOAD(g_SoftNeighborMeta, 0u);
    if (index >= meta.softSoftCandidateCount)
    {
        return;
    }

    if (CRESSIM_SB_LOAD(g_ContactActiveFlags, index) == 0u)
    {
        return;
    }

    const uint outputIndex = CRESSIM_SB_LOAD(g_ContactActiveOffsets, index);
    CRESSIM_SB_STORE(g_ActiveSoftContacts, outputIndex, CRESSIM_SB_LOAD(g_SoftContacts, index));
}
