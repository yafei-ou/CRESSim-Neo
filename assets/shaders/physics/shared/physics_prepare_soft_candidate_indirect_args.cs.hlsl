#include "../../include/physics/shared/physics_indirect_dispatch.hlsli"
#include "../../include/physics/particle/physics_particle_types.hlsli"

static const uint kComputeThreadGroupSize = 64u;

CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
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

    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectSoftGenerateContacts,
                     MakeArgs(meta.particleParticleCandidateCount));
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectSoftGenerateRigidContacts,
                     MakeArgs(meta.particleRigidCandidateCount));
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectSoftCompactContacts,
                     MakeArgs(meta.particleParticleCandidateCount));
    CRESSIM_SB_STORE(g_PhysicsIndirectDispatchArgs, kPhysicsIndirectSoftCompactRigidContacts,
                     MakeArgs(meta.particleRigidCandidateCount));
}
