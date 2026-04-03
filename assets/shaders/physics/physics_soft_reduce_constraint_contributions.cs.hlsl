#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_SoftConstraintContributionOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftConstraintContributionCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftConstraintContributionIndices);
CRESSIM_STRUCTURED_BUFFER(int4, g_SoftConstraintContributions);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftPositionCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= softParticleCount)
    {
        return;
    }

    int4 accumulated = CRESSIM_SB_LOAD(g_SoftPositionCorrections, particleIndex);
    const uint offset = CRESSIM_SB_LOAD(g_SoftConstraintContributionOffsets, particleIndex);
    const uint count = CRESSIM_SB_LOAD(g_SoftConstraintContributionCounts, particleIndex);

    [loop]
    for (uint i = 0u; i < count; ++i)
    {
        const uint contributionIndex =
            CRESSIM_SB_LOAD(g_SoftConstraintContributionIndices, offset + i);
        accumulated += CRESSIM_SB_LOAD(g_SoftConstraintContributions, contributionIndex);
    }

    CRESSIM_SB_STORE(g_SoftPositionCorrections, particleIndex, accumulated);
}
