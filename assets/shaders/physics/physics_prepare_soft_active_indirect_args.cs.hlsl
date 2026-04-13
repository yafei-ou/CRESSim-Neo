#include "physics/include/physics_rigid_common.hlsli"

static const uint kComputeThreadGroupSize = 64u;

CRESSIM_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);
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

    const GpuSoftNeighborMeta meta = CRESSIM_SB_LOAD(g_SoftNeighborMeta, 0u);
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectSoftSolveContacts,
                     MakeArgs(meta.activeSoftContactCount));
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectSoftSolveRigidContacts,
                     MakeArgs(meta.activeSoftRigidContactCount));
}
