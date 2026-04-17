#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_ParticleCellRangeStartFlags);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= softParticleCount)
    {
        return;
    }

    const uint cellKey = CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, idx).mortonCode;
    bool isRangeStart = false;
    if (idx == 0u)
    {
        isRangeStart = true;
    }
    else
    {
        isRangeStart =
            CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, idx - 1u).mortonCode != cellKey;
    }
    CRESSIM_SB_STORE(g_ParticleCellRangeStartFlags, idx, isRangeStart ? 1u : 0u);
}
