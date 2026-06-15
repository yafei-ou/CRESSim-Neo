#include "../../include/physics/shared/physics_indirect_dispatch.hlsli"
#include "../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"

static const uint kComputeThreadGroupSize = 64u;

CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);
CRESSIM_STRUCTURED_BUFFER(GpuNarrowPhaseMeta, g_NarrowPhaseMeta);
CRESSIM_STRUCTURED_BUFFER(GpuProxyRigidContactMeta, g_ProxyRigidContactMeta);
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

    const GpuBroadPhaseMeta broadPhaseMeta = CRESSIM_SB_LOAD(g_BroadPhaseMeta, 0u);
    const GpuNarrowPhaseMeta narrowPhaseMeta = CRESSIM_SB_LOAD(g_NarrowPhaseMeta, 0u);
    const GpuProxyRigidContactMeta proxyMeta = CRESSIM_SB_LOAD(g_ProxyRigidContactMeta, 0u);
    const uint contactSlotCount =
        broadPhaseMeta.candidatePairCount * kRigidContactsPerPair + proxyMeta.activeContactCount;

    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectRigidGenerateContacts,
                     MakeArgs(narrowPhaseMeta.chunkCount));
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectRigidSolveContacts,
                     MakeArgs(contactSlotCount));
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectRigidSolveContactVelocities,
                     MakeArgs(contactSlotCount));
}
