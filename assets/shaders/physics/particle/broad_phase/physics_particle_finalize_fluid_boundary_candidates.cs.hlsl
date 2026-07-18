#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    uint requiredCount = 0u;
    if (particleCount > 0u)
    {
        const uint lastIndex = particleCount - 1u;
        requiredCount = CRESSIM_SB_LOAD(g_CandidateOffsets, lastIndex) +
                        CRESSIM_SB_LOAD(g_CandidateCounts, lastIndex);
    }

    GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    meta.requiredFluidBoundaryCandidateCount = requiredCount;
    meta.fluidBoundaryCandidateCount = min(requiredCount, fluidBoundaryCandidatePairCapacity);
    meta.fluidBoundaryCandidateOverflow =
        requiredCount > fluidBoundaryCandidatePairCapacity ? 1u : 0u;
    CRESSIM_SB_STORE(g_ParticleNeighborMeta, 0u, meta);
}
