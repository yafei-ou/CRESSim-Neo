#ifndef CRESSIM_NEO_PHYSICS_FLUID_NEIGHBOR_BUILD_HLSLI
#define CRESSIM_NEO_PHYSICS_FLUID_NEIGHBOR_BUILD_HLSLI

#include "physics/fluid/physics_fluid_common.hlsli"

GpuParticleCellRange FindParticleCellRange(uint targetKey)
{
    GpuParticleCellRange missingRange;
    missingRange.cellKey = kInvalidIndex;
    missingRange.startIndex = 0u;
    missingRange.endIndex = 0u;
    missingRange.reserved0 = 0u;

    if (particleCellRangeCapacity == 0u)
    {
        return missingRange;
    }

    uint lo = 0u;
    uint hi = particleCellRangeCapacity;
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

    if (lo < particleCellRangeCapacity)
    {
        const GpuParticleCellRange range = CRESSIM_SB_LOAD(g_ParticleCellRanges, lo);
        if (range.cellKey == targetKey)
        {
            return range;
        }
    }

    return missingRange;
}

bool ShouldProcessFluidNeighbor(uint selfIndex, uint selfEnvironment, uint selfLayer, uint selfMask,
                                uint otherIndex)
{
    if (otherIndex == selfIndex)
    {
        return false;
    }

    if (CRESSIM_SB_LOAD(g_ParticleKinds, otherIndex) != kParticleKindFluid)
    {
        return false;
    }

    const uint4 otherMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, otherIndex);
    if (otherMetadata.x != selfEnvironment)
    {
        return false;
    }

    const uint otherLayer = otherMetadata.z;
    const uint otherMask = otherMetadata.w;
    return (selfMask & otherLayer) != 0u && (otherMask & selfLayer) != 0u;
}

#endif // CRESSIM_NEO_PHYSICS_FLUID_NEIGHBOR_BUILD_HLSLI
