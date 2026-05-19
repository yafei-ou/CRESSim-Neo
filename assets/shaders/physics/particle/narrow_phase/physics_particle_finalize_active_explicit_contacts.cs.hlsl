#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    uint count = 0u;
    if (meta.particleParticleCandidateCount > 0u)
    {
        const uint lastIndex = meta.particleParticleCandidateCount - 1u;
        count = CRESSIM_SB_LOAD(g_ContactActiveOffsets, lastIndex) +
                CRESSIM_SB_LOAD(g_ContactActiveFlags, lastIndex);
    }

    meta.activeParticleContactCount = count;
    CRESSIM_SB_STORE(g_ParticleNeighborMeta, 0u, meta);
}
