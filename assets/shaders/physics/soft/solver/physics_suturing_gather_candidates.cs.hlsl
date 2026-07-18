#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SuturingNeighborLinks);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_ParticleCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_SuturingCandidateCounts);
CRESSIM_RW_STRUCTURED_BUFFER(uint4, g_SuturingCandidateParticles);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint candidateIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (candidateIndex >= meta.particleParticleCandidateCount)
    {
        return;
    }

    const GpuParticleCandidatePair candidate =
        CRESSIM_SB_LOAD(g_ParticleCandidatePairs, candidateIndex);
    uint suturingCompactIndex = kInvalidSuturingIndex;
    uint softParticleIndex = kInvalidSuturingIndex;

    const uint4 linksA = CRESSIM_SB_LOAD(g_SuturingNeighborLinks, candidate.indexA);
    const uint4 linksB = CRESSIM_SB_LOAD(g_SuturingNeighborLinks, candidate.indexB);
    const bool aSuturing = linksA.z != kInvalidSuturingIndex;
    const bool bSuturing = linksB.z != kInvalidSuturingIndex;
    if (aSuturing == bSuturing)
    {
        return;
    }

    if (aSuturing)
    {
        if (CRESSIM_SB_LOAD(g_ParticleOwnerTypes, candidate.indexB) != kParticleOwnerTypeSoftBody)
        {
            return;
        }
        suturingCompactIndex = linksA.z;
        softParticleIndex = candidate.indexB;
    }
    else
    {
        if (CRESSIM_SB_LOAD(g_ParticleOwnerTypes, candidate.indexA) != kParticleOwnerTypeSoftBody)
        {
            return;
        }
        suturingCompactIndex = linksB.z;
        softParticleIndex = candidate.indexA;
    }

    uint slot = 0u;
    InterlockedAdd(g_SuturingCandidateCounts[suturingCompactIndex], 1u, slot);
    if (slot >= maxSuturingCandidatesPerParticle)
    {
        return;
    }

    CRESSIM_SB_STORE(g_SuturingCandidateParticles,
                     suturingCompactIndex * maxSuturingCandidatesPerParticle + slot,
                     uint4(softParticleIndex, kInvalidSuturingIndex,
                           kInvalidSuturingIndex, kInvalidSuturingIndex));
}
