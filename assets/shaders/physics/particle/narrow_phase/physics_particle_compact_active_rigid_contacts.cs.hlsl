#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleRigidContact, g_ParticleRigidContacts);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);
CRESSIM_STRUCTURED_BUFFER(uint, g_ContactActiveOffsets);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleRigidContact, g_ActiveSoftRigidContacts);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (index >= meta.particleRigidCandidateCount)
    {
        return;
    }

    if (CRESSIM_SB_LOAD(g_ContactActiveFlags, index) == 0u)
    {
        return;
    }

    const uint outputIndex = CRESSIM_SB_LOAD(g_ContactActiveOffsets, index);
    CRESSIM_SB_STORE(g_ActiveSoftRigidContacts, outputIndex,
                     CRESSIM_SB_LOAD(g_ParticleRigidContacts, index));
}
