#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_BROAD_PHASE_QUERY_HLSLI
#define CRESSIM_NEO_PHYSICS_PARTICLE_BROAD_PHASE_QUERY_HLSLI

#include "../physics_particle_dispatch_constants.hlsli"
#include "physics_particle_grid.hlsli"
#include "physics_particle_types.hlsli"

GpuParticleCellRange FindCellRange(uint targetKey)
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

bool IsAdjacentSoftParticle(uint particleIndex, uint candidateIndex)
{
    const uint neighborOffset = CRESSIM_SB_LOAD(g_ParticleAdjacencyOffsets, particleIndex);
    const uint neighborCount = CRESSIM_SB_LOAD(g_ParticleAdjacencyCounts, particleIndex);
    [loop]
    for (uint i = 0u; i < neighborCount; ++i)
    {
        if (CRESSIM_SB_LOAD(g_ParticleAdjacencyIndices, neighborOffset + i) == candidateIndex)
        {
            return true;
        }
    }
    return false;
}

bool IsValidSoftSoftCandidate(uint softIndex, float3 softPosition, float softRadius,
                              uint softKind, uint softEnvironment, uint softPhase, uint softLayer, uint softMask,
                              GpuParticleBroadPhaseEntry candidateEntry, out uint otherSoftIndex)
{
    otherSoftIndex = candidateEntry.particleIndex;
    if (candidateEntry.particleType != kParticleBroadPhaseEntryTypeSoft ||
        otherSoftIndex <= softIndex)
    {
        return false;
    }

    const uint4 otherMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, otherSoftIndex);
    const uint otherEnvironment = otherMetadata.x;
    if (otherEnvironment != softEnvironment)
    {
        return false;
    }

    const uint otherPhase = otherMetadata.y;
    const uint otherLayer = otherMetadata.z;
    const uint otherMask = otherMetadata.w;
    if ((softMask & otherLayer) == 0u || (otherMask & softLayer) == 0u)
    {
        return false;
    }

    const uint otherKind = CRESSIM_SB_LOAD(g_ParticleKinds, otherSoftIndex);
    if (softKind == kParticleKindSoftSolid && otherKind == kParticleKindSoftSolid &&
        ParticlePhaseGroup(softPhase) == ParticlePhaseGroup(otherPhase))
    {
        const bool selfCollideA = ParticlePhaseSelfCollideEnabled(softPhase);
        const bool selfCollideB = ParticlePhaseSelfCollideEnabled(otherPhase);
        if (!selfCollideA || !selfCollideB || IsAdjacentSoftParticle(softIndex, otherSoftIndex))
        {
            return false;
        }
    }

    const float3 otherPosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, otherSoftIndex).xyz;
    const float combinedRadius = softRadius + CRESSIM_SB_LOAD(g_ParticleRadii, otherSoftIndex);
    const float3 deltaPos = otherPosition - softPosition;
    return dot(deltaPos, deltaPos) <= combinedRadius * combinedRadius;
}

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_BROAD_PHASE_QUERY_HLSLI
