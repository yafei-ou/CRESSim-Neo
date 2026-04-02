#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftRigidCellRange, g_SoftRigidCellRanges);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= softRigidCellRangeCapacity)
    {
        return;
    }

    GpuSoftRigidCellRange range;
    range.cellKey = kInvalidIndex;
    range.startIndex = 0u;
    range.endIndex = 0u;
    range.reserved0 = 0u;
    CRESSIM_SB_STORE(g_SoftRigidCellRanges, idx, range);
}
