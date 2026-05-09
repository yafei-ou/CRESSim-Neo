#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleContact, g_ParticleContacts);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveOffsets);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleContact, g_ActiveSoftContacts);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (index >= meta.particleParticleCandidateCount)
    {
        return;
    }

    if (CRESSIM_SB_LOAD(g_ContactActiveFlags, index) == 0u)
    {
        return;
    }

    const uint outputIndex = CRESSIM_SB_LOAD(g_ContactActiveOffsets, index);
    CRESSIM_SB_STORE(g_ActiveSoftContacts, outputIndex, CRESSIM_SB_LOAD(g_ParticleContacts, index));
}
