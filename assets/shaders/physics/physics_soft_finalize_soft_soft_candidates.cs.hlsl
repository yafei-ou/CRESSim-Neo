#include "include/physics/physics_soft_dispatch_constants.hlsli"
#include "include/physics/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    uint requiredCount = 0u;
    if (softParticleCount > 0u)
    {
        const uint lastIndex = softParticleCount - 1u;
        requiredCount = CRESSIM_SB_LOAD(g_CandidateOffsets, lastIndex) +
                        CRESSIM_SB_LOAD(g_CandidateCounts, lastIndex);
    }

    GpuSoftNeighborMeta meta = CRESSIM_SB_LOAD(g_SoftNeighborMeta, 0u);
    meta.requiredSoftSoftCandidateCount = requiredCount;
    meta.softSoftCandidateCount = min(requiredCount, softCandidatePairCapacity);
    meta.softSoftCandidateOverflow = requiredCount > softCandidatePairCapacity ? 1u : 0u;
    CRESSIM_SB_STORE(g_SoftNeighborMeta, 0u, meta);
}
