cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint activeDynamicCount;
    uint candidatePairCount;
    uint candidatePairCapacity;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
};

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_PairCountsSphereSphere;
StructuredBuffer<uint> g_PairCountsSphereBox;
StructuredBuffer<uint> g_PairCountsSphereCapsule;
StructuredBuffer<uint> g_PairCountsBoxBox;
StructuredBuffer<uint> g_PairCountsBoxCapsule;
StructuredBuffer<uint> g_PairCountsCapsuleCapsule;
StructuredBuffer<uint> g_PairOffsetsSphereSphere;
StructuredBuffer<uint> g_PairOffsetsSphereBox;
StructuredBuffer<uint> g_PairOffsetsSphereCapsule;
StructuredBuffer<uint> g_PairOffsetsBoxBox;
StructuredBuffer<uint> g_PairOffsetsBoxCapsule;
StructuredBuffer<uint> g_PairOffsetsCapsuleCapsule;
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
    if (activeDynamicCount > 0u)
    {
        const uint lastIndex = activeDynamicCount - 1u;
        counts[0] = g_PairOffsetsSphereSphere[lastIndex] + g_PairCountsSphereSphere[lastIndex];
        counts[1] = g_PairOffsetsSphereBox[lastIndex] + g_PairCountsSphereBox[lastIndex];
        counts[2] = g_PairOffsetsSphereCapsule[lastIndex] + g_PairCountsSphereCapsule[lastIndex];
        counts[3] = g_PairOffsetsBoxBox[lastIndex] + g_PairCountsBoxBox[lastIndex];
        counts[4] = g_PairOffsetsBoxCapsule[lastIndex] + g_PairCountsBoxCapsule[lastIndex];
        counts[5] = g_PairOffsetsCapsuleCapsule[lastIndex] + g_PairCountsCapsuleCapsule[lastIndex];
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
