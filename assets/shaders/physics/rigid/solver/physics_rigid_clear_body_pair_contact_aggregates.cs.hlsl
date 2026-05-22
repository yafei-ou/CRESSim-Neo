#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"

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

    GpuRigidBodyPairContactAggregateHeader header;
    header.bodyA = 0xffffffffu;
    header.bodyB = 0xffffffffu;
    header.count = 0u;
    header.flags = 0u;
    CRESSIM_SB_STORE(g_RigidBodyPairAggregateHeaders, aggregateIndex, header);
}
