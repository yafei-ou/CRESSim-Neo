#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuSoftRigidBroadPhaseParticle, g_SoftRigidBroadPhaseParticles);
CRESSIM_RW_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SoftRigidBroadPhaseKeys);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    const uint totalParticleCount = softParticleCount + rigidSurfaceParticleCount;
    if (idx >= totalParticleCount)
    {
        return;
    }

    const GpuSoftRigidBroadPhaseParticle entry =
        CRESSIM_SB_LOAD(g_SoftRigidBroadPhaseParticles, idx);
    GpuMortonCodeElement key;
    key.mortonCode = entry.cellKey;
    key.elementIdx = idx;
    CRESSIM_SB_STORE(g_SoftRigidBroadPhaseKeys, idx, key);
}
