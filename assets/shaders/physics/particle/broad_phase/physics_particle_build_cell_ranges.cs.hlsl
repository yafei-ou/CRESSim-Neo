#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleCellRangeStartFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleCellRangeStartOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= particleCount)
    {
        return;
    }

    const GpuMortonCodeElement keyEntry = CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, idx);
    const uint cellKey = keyEntry.mortonCode;

    if (CRESSIM_SB_LOAD(g_ParticleCellRangeStartFlags, idx) == 0u)
    {
        return;
    }

    uint endIndex = idx + 1u;
    [loop]
    while (true)
    {
        if (endIndex >= particleCount)
        {
            break;
        }

        const uint nextCellKey =
            CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, endIndex).mortonCode;
        if (nextCellKey != cellKey)
        {
            break;
        }

        ++endIndex;
    }

    const uint rangeIndex = CRESSIM_SB_LOAD(g_ParticleCellRangeStartOffsets, idx);
    if (rangeIndex >= particleCellRangeCapacity)
    {
        return;
    }

    GpuParticleCellRange range;
    range.cellKey = cellKey;
    range.startIndex = idx;
    range.endIndex = endIndex;
    range.reserved0 = 0u;
    CRESSIM_SB_STORE(g_ParticleCellRanges, rangeIndex, range);
}
