#ifndef CRESSIM_NEO_PHYSICS_SOFT_CELL_RANGE_QUERY_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_CELL_RANGE_QUERY_HLSLI

#include "../physics_soft_dispatch_constants.hlsli"
#include "physics_soft_grid.hlsli"

GpuParticleCellRange FindCellRange(uint targetKey)
{
    GpuParticleCellRange missingRange;
    missingRange.cellKey = kInvalidIndex;
    missingRange.startIndex = 0u;
    missingRange.endIndex = 0u;
    missingRange.reserved0 = 0u;

    if (softCellRangeCapacity == 0u)
    {
        return missingRange;
    }

    uint lo = 0u;
    uint hi = softCellRangeCapacity;
    [loop]
    while (lo < hi)
    {
        const uint mid = lo + (hi - lo) / 2u;
        const GpuParticleCellRange range = CRESSIM_SB_LOAD(g_ParticleCellRanges, mid);
        if (range.cellKey < targetKey)
        {
            lo = mid + 1u;
        }
        else
        {
            hi = mid;
        }
    }

    if (lo < softCellRangeCapacity)
    {
        const GpuParticleCellRange range = CRESSIM_SB_LOAD(g_ParticleCellRanges, lo);
        if (range.cellKey == targetKey)
        {
            return range;
        }
    }

    return missingRange;
}

#endif // CRESSIM_NEO_PHYSICS_SOFT_CELL_RANGE_QUERY_HLSLI
