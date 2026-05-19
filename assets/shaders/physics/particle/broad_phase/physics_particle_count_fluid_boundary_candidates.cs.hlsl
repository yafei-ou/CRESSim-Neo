#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_BvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_StaticBvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);

CRESSIM_RW_STRUCTURED_BUFFER(uint, g_CandidateCounts);

float ComputeFluidBoundaryQueryRadius(uint particleIndex)
{
    const float particleRadius = max(CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex), 0.0);
    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float supportRadius = max(fluidMaterial.smoothingRadius, particleRadius);
    const float reuseMargin = max(max(fluidMaterial.cflRadius, particleRadius), supportRadius);
    return supportRadius + reuseMargin;
}

void CountCandidatesFromBvh(bool useStaticBvh, float3 queryMin, float3 queryMax,
                            uint softEnvironment, uint softLayer, uint softMask,
                            inout uint count)
{
    if (rigidColliderCount == 0u)
    {
        return;
    }

    uint stack[128];
    uint stackSize = 0u;
    stack[stackSize++] = 0u;

    while (stackSize > 0u)
    {
        const uint nodeIndex = stack[--stackSize];
        GpuBvhNode node;
        if (useStaticBvh)
        {
            node = CRESSIM_SB_LOAD(g_StaticBvhNodes, nodeIndex);
        }
        else
        {
            node = CRESSIM_SB_LOAD(g_BvhNodes, nodeIndex);
        }
        if (!NodeAabbOverlapsQuery(node, queryMin, queryMax))
        {
            continue;
        }

        if (node.left < 0 && node.right < 0)
        {
            const uint colliderIndex = node.primitiveIdx;
            const GpuColliderBroadPhaseData collider =
                CRESSIM_SB_LOAD(g_ColliderBroadPhaseData, colliderIndex);
            const uint rigidBodyIndex = collider.ownerBody;
            const uint rigidBodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndex);
            if (collider.enabledFlag != 0u && collider.environmentIndex == softEnvironment &&
                rigidBodyType != kRigidBodyTypeDynamic &&
                (softMask & collider.collisionLayer) != 0u &&
                (collider.collisionMask & softLayer) != 0u)
            {
                ++count;
            }
            continue;
        }

        if (node.left >= 0 && stackSize < 128u)
        {
            stack[stackSize++] = (uint)node.left;
        }
        if (node.right >= 0 && stackSize < 128u)
        {
            stack[stackSize++] = (uint)node.right;
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    if (CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        CRESSIM_SB_STORE(g_CandidateCounts, particleIndex, 0u);
        return;
    }

    const float3 particlePosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const uint4 particleMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, particleIndex);
    const uint particleEnvironment = particleMetadata.x;
    const uint particleLayer = particleMetadata.z;
    const uint particleMask = particleMetadata.w;

    uint count = 0u;
    const float queryRadius = ComputeFluidBoundaryQueryRadius(particleIndex);
    const float3 queryExtent = float3(queryRadius, queryRadius, queryRadius);
    const float3 queryMin = particlePosition - queryExtent;
    const float3 queryMax = particlePosition + queryExtent;
    const GpuBroadPhaseMeta broadPhaseMeta = CRESSIM_SB_LOAD(g_BroadPhaseMeta, 0u);

    if (broadPhaseMeta.activeMovingCount > 0u)
    {
        CountCandidatesFromBvh(false, queryMin, queryMax, particleEnvironment, particleLayer,
                               particleMask, count);
    }
    CountCandidatesFromBvh(true, queryMin, queryMax, particleEnvironment, particleLayer,
                           particleMask, count);

    CRESSIM_SB_STORE(g_CandidateCounts, particleIndex, count);
}
