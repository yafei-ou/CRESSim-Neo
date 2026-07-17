#include "physics/shared/physics_indirect_dispatch.hlsli"

static const uint kComputeThreadGroupSize = 64u;

CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyPairAggregateActiveCount);
CRESSIM_RW_STRUCTURED_BUFFER(GpuDispatchIndirectArgs, g_PhysicsIndirectDispatchArgs);

GpuDispatchIndirectArgs MakeArgs(uint count)
{
    GpuDispatchIndirectArgs args;
    args.groupCountX = max(1u, (count + kComputeThreadGroupSize - 1u) / kComputeThreadGroupSize);
    args.groupCountY = 1u;
    args.groupCountZ = 1u;
    return args;
}

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    const uint activePairCount = CRESSIM_SB_LOAD(g_RigidBodyPairAggregateActiveCount, 0u);
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectRigidSolveContactVelocities,
                     MakeArgs(activePairCount));
}
