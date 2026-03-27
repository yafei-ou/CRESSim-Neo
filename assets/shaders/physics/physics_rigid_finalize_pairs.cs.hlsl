#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_PairCountsSphereSphere);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairCountsSphereBox);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairCountsSphereCapsule);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairCountsBoxBox);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairCountsBoxCapsule);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairCountsCapsuleCapsule);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsSphereSphere);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsSphereBox);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsSphereCapsule);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsBoxBox);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsBoxCapsule);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsCapsuleCapsule);
RWStructuredBuffer<GpuRigidPairRange> g_RigidPairRanges;
RWStructuredBuffer<GpuBroadPhaseMeta> g_BroadPhaseMeta;

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    uint counts[kRigidPairTypeCount];
    [unroll] for (uint i = 0u; i < kRigidPairTypeCount; ++i)
    {
        counts[i] = 0u;
    }
    if (activeMovingCount > 0u)
    {
        const uint lastIndex = activeMovingCount - 1u;
        counts[0] = CRESSIM_SB_LOAD(g_PairOffsetsSphereSphere, lastIndex) +
                    CRESSIM_SB_LOAD(g_PairCountsSphereSphere, lastIndex);
        counts[1] = CRESSIM_SB_LOAD(g_PairOffsetsSphereBox, lastIndex) +
                    CRESSIM_SB_LOAD(g_PairCountsSphereBox, lastIndex);
        counts[2] = CRESSIM_SB_LOAD(g_PairOffsetsSphereCapsule, lastIndex) +
                    CRESSIM_SB_LOAD(g_PairCountsSphereCapsule, lastIndex);
        counts[3] = CRESSIM_SB_LOAD(g_PairOffsetsBoxBox, lastIndex) +
                    CRESSIM_SB_LOAD(g_PairCountsBoxBox, lastIndex);
        counts[4] = CRESSIM_SB_LOAD(g_PairOffsetsBoxCapsule, lastIndex) +
                    CRESSIM_SB_LOAD(g_PairCountsBoxCapsule, lastIndex);
        counts[5] = CRESSIM_SB_LOAD(g_PairOffsetsCapsuleCapsule, lastIndex) +
                    CRESSIM_SB_LOAD(g_PairCountsCapsuleCapsule, lastIndex);
    }

    uint runningStart = 0u;
    [unroll] for (uint type = 0u; type < kRigidPairTypeCount; ++type)
    {
        GpuRigidPairRange range;
        range.type = type;
        range.start = runningStart;
        range.count = counts[type];
        range.reserved = 0u;
        g_RigidPairRanges[type] = range;
        runningStart += counts[type];
    }

    GpuBroadPhaseMeta meta = g_BroadPhaseMeta[0];
    meta.candidatePairCount = runningStart;
    meta.requiredPairCount = runningStart;
    meta.overflow = runningStart > candidatePairCapacity ? 1u : 0u;
    g_BroadPhaseMeta[0] = meta;
}
