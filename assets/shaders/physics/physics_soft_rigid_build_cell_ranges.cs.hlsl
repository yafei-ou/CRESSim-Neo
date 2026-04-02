#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedSoftRigidBroadPhaseKeys);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftRigidCellRange, g_SoftRigidCellRanges);

uint FindCellRangeSlot(uint cellKey)
{
    const uint mask = softRigidCellRangeCapacity - 1u;
    uint slot = cellKey & mask;
    [loop]
    for (uint probe = 0u; probe < softRigidCellRangeCapacity; ++probe)
    {
        uint originalKey = kInvalidIndex;
        InterlockedCompareExchange(CRESSIM_SB_REF(g_SoftRigidCellRanges, slot).cellKey,
                                   kInvalidIndex, cellKey, originalKey);
        if (originalKey == kInvalidIndex || originalKey == cellKey)
        {
            return slot;
        }

        slot = (slot + 1u) & mask;
    }

    return kInvalidIndex;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    const uint totalParticleCount = softParticleCount + rigidSurfaceParticleCount;
    if (idx >= totalParticleCount)
    {
        return;
    }

    const GpuMortonCodeElement keyEntry = CRESSIM_SB_LOAD(g_SortedSoftRigidBroadPhaseKeys, idx);
    const uint cellKey = keyEntry.mortonCode;

    const bool isRangeStart =
        idx == 0u || CRESSIM_SB_LOAD(g_SortedSoftRigidBroadPhaseKeys, idx - 1u).mortonCode != cellKey;
    if (!isRangeStart)
    {
        return;
    }

    uint endIndex = idx + 1u;
    while (endIndex < totalParticleCount &&
           CRESSIM_SB_LOAD(g_SortedSoftRigidBroadPhaseKeys, endIndex).mortonCode == cellKey)
    {
        ++endIndex;
    }

    const uint slot = FindCellRangeSlot(cellKey);
    if (slot == kInvalidIndex)
    {
        return;
    }

    GpuSoftRigidCellRange range;
    range.cellKey = cellKey;
    range.startIndex = idx;
    range.endIndex = endIndex;
    range.reserved0 = 0u;
    CRESSIM_SB_STORE(g_SoftRigidCellRanges, slot, range);
}
