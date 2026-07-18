#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(uint, g_SuturingCandidateCounts);
CRESSIM_RW_STRUCTURED_BUFFER(uint4, g_SuturingCandidateParticles);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint compactIndex = dispatchThreadID.x;
    if (compactIndex >= suturingParticleCount)
    {
        return;
    }

    CRESSIM_SB_STORE(g_SuturingCandidateCounts, compactIndex, 0u);

    const uint baseIndex = compactIndex * maxSuturingCandidatesPerParticle;
    [loop]
    for (uint candidateIndex = 0u; candidateIndex < maxSuturingCandidatesPerParticle; ++candidateIndex)
    {
        CRESSIM_SB_STORE(g_SuturingCandidateParticles, baseIndex + candidateIndex,
                         uint4(kInvalidSuturingIndex, kInvalidSuturingIndex,
                               kInvalidSuturingIndex, kInvalidSuturingIndex));
    }
}
