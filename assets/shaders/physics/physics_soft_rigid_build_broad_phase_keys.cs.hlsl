#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_RW_STRUCTURED_BUFFER(GpuMortonCodeElement, g_ParticleBroadPhaseKeys);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= softParticleCount)
    {
        return;
    }

    const GpuParticleBroadPhaseEntry entry =
        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, idx);
    GpuMortonCodeElement key;
    key.mortonCode = entry.cellKey;
    key.elementIdx = idx;
    CRESSIM_SB_STORE(g_ParticleBroadPhaseKeys, idx, key);
}
