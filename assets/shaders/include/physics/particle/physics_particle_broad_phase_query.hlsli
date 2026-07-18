#ifndef CRESSIM_NEO_PHYSICS_PARTICLE_BROAD_PHASE_QUERY_HLSLI
#define CRESSIM_NEO_PHYSICS_PARTICLE_BROAD_PHASE_QUERY_HLSLI

#include "physics_particle_dispatch_constants.hlsli"
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

bool IsAdjacentParticle(uint particleIndex, uint candidateIndex)
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

bool IsValidParticleParticleCandidate(uint particleIndex, float3 particlePosition, float particleRadius,
                                      uint particleKind, uint particleEnvironment,
                                      uint particlePhase, uint particleLayer, uint particleMask,
                                      GpuParticleBroadPhaseEntry candidateEntry,
                                      out uint otherParticleIndex)
{
    otherParticleIndex = candidateEntry.particleIndex;
    if (candidateEntry.particleType != kParticleBroadPhaseEntryTypeParticle ||
        otherParticleIndex <= particleIndex)
    {
        return false;
    }

    const uint4 otherMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, otherParticleIndex);
    const uint otherEnvironment = otherMetadata.x;
    if (otherEnvironment != particleEnvironment)
    {
        return false;
    }

    const uint otherPhase = otherMetadata.y;
    const uint otherLayer = otherMetadata.z;
    const uint otherMask = otherMetadata.w;
    if ((particleMask & otherLayer) == 0u || (otherMask & particleLayer) == 0u)
    {
        return false;
    }

    const uint otherKind = CRESSIM_SB_LOAD(g_ParticleKinds, otherParticleIndex);
    if (particleKind == kParticleKindFluid && otherKind == kParticleKindFluid)
    {
        return false;
    }

    if (particleKind == kParticleKindSolid && otherKind == kParticleKindSolid &&
        ParticlePhaseGroup(particlePhase) == ParticlePhaseGroup(otherPhase))
    {
        const bool selfCollideA = ParticlePhaseSelfCollideEnabled(particlePhase);
        const bool selfCollideB = ParticlePhaseSelfCollideEnabled(otherPhase);
        if (!selfCollideA || !selfCollideB ||
            IsAdjacentParticle(particleIndex, otherParticleIndex))
        {
            return false;
        }
    }

    const float3 otherPosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, otherParticleIndex).xyz;
    const float combinedRadius =
        particleRadius + CRESSIM_SB_LOAD(g_ParticleRadii, otherParticleIndex);
    const float3 deltaPos = otherPosition - particlePosition;
    return dot(deltaPos, deltaPos) <= combinedRadius * combinedRadius;
}

#endif // CRESSIM_NEO_PHYSICS_PARTICLE_BROAD_PHASE_QUERY_HLSLI
