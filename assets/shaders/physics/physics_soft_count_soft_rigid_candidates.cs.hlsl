#include "include/physics/physics_soft_dispatch_constants.hlsli"
#include "include/physics/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_SoftParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SoftParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_BvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_StaticBvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);
CRESSIM_STRUCTURED_BUFFER(uint2, g_BodyColliderRanges);

CRESSIM_RW_STRUCTURED_BUFFER(uint, g_CandidateCounts);

bool NodeOverlapsQuery(GpuBvhNode node, float3 queryMin, float3 queryMax)
{
    return AabbOverlaps(queryMin, queryMax, float3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
                        float3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ));
}

bool TryAppendRigidBody(uint rigidBodyIndex,
                        inout uint seenRigidBodies[kSoftRigidDedupCacheSize],
                        inout uint seenRigidCount)
{
    [unroll]
    for (uint i = 0u; i < kSoftRigidDedupCacheSize; ++i)
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

    if (seenRigidCount < kSoftRigidDedupCacheSize)
    {
        seenRigidBodies[seenRigidCount] = rigidBodyIndex;
        ++seenRigidCount;
    }
    return true;
}

void CountCandidatesFromBvh(bool useStaticBvh, float3 queryMin, float3 queryMax,
                            uint softEnvironment, uint softLayer, uint softMask,
                            inout uint seenRigidBodies[kSoftRigidDedupCacheSize],
                            inout uint seenRigidCount, inout uint count)
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
        if (!NodeOverlapsQuery(node, queryMin, queryMax))
        {
            continue;
        }

        if (node.left < 0 && node.right < 0)
        {
            const uint colliderIndex = node.primitiveIdx;
            const GpuColliderBroadPhaseData collider =
                CRESSIM_SB_LOAD(g_ColliderBroadPhaseData, colliderIndex);
            const uint rigidBodyIndex = collider.ownerBody;
            if (collider.enabledFlag != 0u && collider.environmentIndex == softEnvironment &&
                (softMask & collider.collisionLayer) != 0u &&
                (collider.collisionMask & softLayer) != 0u &&
                CRESSIM_SB_LOAD(g_BodyColliderRanges, rigidBodyIndex).y > 0u &&
                TryAppendRigidBody(rigidBodyIndex, seenRigidBodies, seenRigidCount))
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
    const uint softIndex = dispatchThreadID.x;
    if (softIndex >= softParticleCount)
    {
        return;
    }

    const float3 softPosition = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, softIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_SoftParticleRadii, softIndex);
    const uint4 softMetadata = CRESSIM_SB_LOAD(g_SoftParticleBroadPhaseMetadata, softIndex);
    const uint softEnvironment = softMetadata.x;
    const uint softLayer = softMetadata.z;
    const uint softMask = softMetadata.w;

    uint seenRigidBodies[kSoftRigidDedupCacheSize];
    uint seenRigidCount = 0u;
    uint count = 0u;
    const float3 queryExtent = float3(softRadius, softRadius, softRadius);
    const float3 queryMin = softPosition - queryExtent;
    const float3 queryMax = softPosition + queryExtent;
    const GpuBroadPhaseMeta broadPhaseMeta = CRESSIM_SB_LOAD(g_BroadPhaseMeta, 0u);

    if (broadPhaseMeta.activeMovingCount > 0u)
    {
        CountCandidatesFromBvh(false, queryMin, queryMax, softEnvironment, softLayer, softMask,
                               seenRigidBodies, seenRigidCount, count);
    }
    CountCandidatesFromBvh(true, queryMin, queryMax, softEnvironment, softLayer,
                           softMask, seenRigidBodies, seenRigidCount, count);

    CRESSIM_SB_STORE(g_CandidateCounts, softIndex, count);
}
