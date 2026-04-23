#ifndef CRESSIM_NEO_PHYSICS_SOFT_SOFT_CANDIDATE_QUERY_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_SOFT_CANDIDATE_QUERY_HLSLI

#include "physics_soft_types.hlsli"

bool IsAdjacentSoftParticle(uint particleIndex, uint candidateIndex)
{
    const uint neighborOffset = CRESSIM_SB_LOAD(g_SoftParticleAdjacencyOffsets, particleIndex);
    const uint neighborCount = CRESSIM_SB_LOAD(g_SoftParticleAdjacencyCounts, particleIndex);
    [loop]
    for (uint i = 0u; i < neighborCount; ++i)
    {
        if (CRESSIM_SB_LOAD(g_SoftParticleAdjacencyIndices, neighborOffset + i) == candidateIndex)
        {
            return true;
        }
    }
    return false;
}

bool IsValidSoftSoftCandidate(uint softIndex, float3 softPosition, float softRadius,
                              uint softEnvironment, uint softPhase, uint softLayer, uint softMask,
                              GpuParticleBroadPhaseEntry candidateEntry, out uint otherSoftIndex)
{
    otherSoftIndex = candidateEntry.particleIndex;
    if (candidateEntry.particleType != kParticleBroadPhaseEntryTypeSoft ||
        otherSoftIndex <= softIndex)
    {
        return false;
    }

    const uint4 otherMetadata = CRESSIM_SB_LOAD(g_SoftParticleBroadPhaseMetadata, otherSoftIndex);
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

    if (SoftParticlePhaseGroup(softPhase) == SoftParticlePhaseGroup(otherPhase))
    {
        const bool selfCollideA = SoftParticlePhaseSelfCollideEnabled(softPhase);
        const bool selfCollideB = SoftParticlePhaseSelfCollideEnabled(otherPhase);
        if (!selfCollideA || !selfCollideB || IsAdjacentSoftParticle(softIndex, otherSoftIndex))
        {
            return false;
        }
    }

    const float3 otherPosition = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, otherSoftIndex).xyz;
    const float combinedRadius = softRadius + CRESSIM_SB_LOAD(g_SoftParticleRadii, otherSoftIndex);
    const float3 deltaPos = otherPosition - softPosition;
    return dot(deltaPos, deltaPos) <= combinedRadius * combinedRadius;
}

#endif // CRESSIM_NEO_PHYSICS_SOFT_SOFT_CANDIDATE_QUERY_HLSLI
