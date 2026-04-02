#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuSoftBroadPhaseParticle, g_SoftBroadPhaseParticles);
CRESSIM_RW_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SoftBroadPhaseKeys);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    const uint totalParticleCount = softParticleCount + rigidSurfaceParticleCount;
    if (idx >= totalParticleCount)
    {
        return;
    }

    const GpuSoftBroadPhaseParticle entry = CRESSIM_SB_LOAD(g_SoftBroadPhaseParticles, idx);
    GpuMortonCodeElement key;
    key.mortonCode = entry.cellKey;
    key.elementIdx = idx;
    CRESSIM_SB_STORE(g_SoftBroadPhaseKeys, idx, key);
}
