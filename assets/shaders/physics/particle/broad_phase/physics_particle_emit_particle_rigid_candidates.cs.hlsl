#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_rigid_types.hlsli"
#include "physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_BvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_StaticBvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);
CRESSIM_STRUCTURED_BUFFER(uint2, g_BodyColliderRanges);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);

CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_ParticleCandidatePairs);

bool TryAppendRigidBody(uint rigidBodyIndex,
                        inout uint seenRigidBodies[kParticleRigidDedupCacheSize],
                        inout uint seenRigidCount)
{
    [unroll]
    for (uint i = 0u; i < kParticleRigidDedupCacheSize; ++i)
    {
        if (i >= seenRigidCount)
        {
            break;
        }
        if (seenRigidBodies[i] == rigidBodyIndex)
        {
            return false;
        }
    }

    if (seenRigidCount < kParticleRigidDedupCacheSize)
    {
        seenRigidBodies[seenRigidCount] = rigidBodyIndex;
        ++seenRigidCount;
    }
    return true;
}

void EmitCandidatesFromBvh(bool useStaticBvh, float3 queryMin, float3 queryMax,
                           uint particleKind, uint softEnvironment, uint softLayer,
                           uint softMask, uint softIndex, bool rigidProxyOwner,
                           uint owningRigidBodyIndex,
                           inout uint seenRigidBodies[kParticleRigidDedupCacheSize],
                           inout uint seenRigidCount, inout uint writeIndex)
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
            if (rigidProxyOwner && rigidBodyIndex == owningRigidBodyIndex)
            {
                continue;
            }
            const uint rigidBodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndex);
            if (collider.enabledFlag != 0u && collider.environmentIndex == softEnvironment &&
                (softMask & collider.collisionLayer) != 0u &&
                (collider.collisionMask & softLayer) != 0u &&
                (particleKind == kParticleKindSolid ||
                 rigidBodyType == kRigidBodyTypeDynamic) &&
                CRESSIM_SB_LOAD(g_BodyColliderRanges, rigidBodyIndex).y > 0u &&
                TryAppendRigidBody(rigidBodyIndex, seenRigidBodies, seenRigidCount))
            {
                if (writeIndex >= particleCandidatePairCapacity)
                {
                    return;
                }

                GpuParticleCandidatePair pair;
                pair.pairType = kParticleCandidatePairTypeParticleRigid;
                pair.indexA = softIndex;
                pair.indexB = rigidBodyIndex;
                pair.auxIndex = 0u;
                CRESSIM_SB_STORE(g_ParticleCandidatePairs, writeIndex, pair);
                ++writeIndex;
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
    const uint softIndex = dispatchThreadID.x;
    if (softIndex >= particleCount)
    {
        return;
    }

    const uint candidateCount = CRESSIM_SB_LOAD(g_CandidateCounts, softIndex);
    if (candidateCount == 0u)
    {
        return;
    }

    const float3 softPosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, softIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_ParticleRadii, softIndex);
    const uint particleKind = CRESSIM_SB_LOAD(g_ParticleKinds, softIndex);
    const uint4 softMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, softIndex);
    const uint softEnvironment = softMetadata.x;
    const uint softLayer = softMetadata.z;
    const uint softMask = softMetadata.w;
    const uint ownerType = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, softIndex);
    const bool rigidProxyOwner = ownerType == kParticleOwnerTypeRigidBody;
    const uint owningRigidBodyIndex = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, softIndex);
    uint seenRigidBodies[kParticleRigidDedupCacheSize];
    uint seenRigidCount = 0u;
    uint writeIndex = CRESSIM_SB_LOAD(g_CandidateOffsets, softIndex);
    const float3 queryExtent = float3(softRadius, softRadius, softRadius);
    const float3 queryMin = softPosition - queryExtent;
    const float3 queryMax = softPosition + queryExtent;
    const GpuBroadPhaseMeta broadPhaseMeta = CRESSIM_SB_LOAD(g_BroadPhaseMeta, 0u);

    if (broadPhaseMeta.activeMovingCount > 0u)
    {
        EmitCandidatesFromBvh(false, queryMin, queryMax, particleKind, softEnvironment,
                              softLayer, softMask, softIndex, rigidProxyOwner,
                              owningRigidBodyIndex, seenRigidBodies,
                              seenRigidCount, writeIndex);
    }
    EmitCandidatesFromBvh(true, queryMin, queryMax, particleKind, softEnvironment, softLayer,
                          softMask, softIndex, rigidProxyOwner, owningRigidBodyIndex,
                          seenRigidBodies, seenRigidCount, writeIndex);
}
