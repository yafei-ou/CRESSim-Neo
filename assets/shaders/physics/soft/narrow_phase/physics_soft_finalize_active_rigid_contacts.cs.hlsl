#include "../../../include/physics/soft/physics_soft_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuSoftNeighborMeta meta = CRESSIM_SB_LOAD(g_SoftNeighborMeta, 0u);
    uint count = 0u;
    if (meta.softRigidCandidateCount > 0u)
    {
        const uint lastIndex = meta.softRigidCandidateCount - 1u;
        count = CRESSIM_SB_LOAD(g_ContactActiveOffsets, lastIndex) +
                CRESSIM_SB_LOAD(g_ContactActiveFlags, lastIndex);
    }

    meta.activeSoftRigidContactCount = count;
    CRESSIM_SB_STORE(g_SoftNeighborMeta, 0u, meta);
}
