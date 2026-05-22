#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidBodyPairContactAggregateMapEntry,
                             g_RigidBodyPairAggregateMap);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RigidBodyPairAggregateActiveCount);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidBodyPairContactAggregateHeader,
                             g_RigidBodyPairAggregateHeaders);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint aggregateIndex = dispatchThreadID.x;
    if (aggregateIndex >= candidatePairCapacity)
    {
        return;
    }

    GpuRigidBodyPairContactAggregateMapEntry mapEntry;
    mapEntry.bodyA = kRigidInvalidAggregateIndex;
    mapEntry.bodyB = kRigidInvalidAggregateIndex;
    mapEntry.pairIndex = kRigidInvalidAggregateIndex;
    mapEntry.flags = 0u;
    CRESSIM_SB_STORE(g_RigidBodyPairAggregateMap, aggregateIndex, mapEntry);

    GpuRigidBodyPairContactAggregateHeader header;
    header.bodyA = kRigidInvalidAggregateIndex;
    header.bodyB = kRigidInvalidAggregateIndex;
    header.count = 0u;
    header.flags = 0u;
    CRESSIM_SB_STORE(g_RigidBodyPairAggregateHeaders, aggregateIndex, header);

    if (aggregateIndex == 0u)
    {
        CRESSIM_SB_STORE(g_RigidBodyPairAggregateActiveCount, 0u, 0u);
    }
}
