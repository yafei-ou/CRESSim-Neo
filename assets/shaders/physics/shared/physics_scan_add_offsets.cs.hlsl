#include "physics/core/physics_base.hlsli"
#include "physics/physics_scan_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ScannedBlockOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_ScanOutput);

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID,
                                 uint3 groupID : SV_GroupID)
{
    if (!ScanHasParentOffsets())
    {
        return;
    }

    const uint index = dispatchThreadID.x;
    if (index >= ScanElementCount())
    {
        return;
    }

    CRESSIM_SB_REF(g_ScanOutput, index) += CRESSIM_SB_LOAD(g_ScannedBlockOffsets, groupID.x);
}
