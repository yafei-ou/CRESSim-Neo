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

StructuredBuffer<uint> g_ActiveBodyIndices;
StructuredBuffer<GpuBodyAabb> g_BodyAabbs;
StructuredBuffer<GpuBodyMeta> g_BodyMeta;
StructuredBuffer<GpuBvhNode> g_BvhNodes;
RWStructuredBuffer<uint> g_PairCounts;

bool NodeOverlapsQuery(GpuBvhNode node, float3 queryMin, float3 queryMax)
{
    return AabbOverlaps(queryMin, queryMax, float3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
                        float3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ));
}

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint activeIndex = dispatchThreadID.x;
    if (activeIndex >= activeDynamicCount)
    {
        return;
    }

    const uint bodyId = g_ActiveBodyIndices[activeIndex];
    const GpuBodyAabb bodyAabb = g_BodyAabbs[bodyId];
    const float3 queryMin = bodyAabb.minBounds.xyz;
    const float3 queryMax = bodyAabb.maxBounds.xyz;

    uint pairCount = 0u;

    if (activeDynamicCount > 1u)
    {
        uint stack[128];
        uint stackSize = 0u;
        stack[stackSize++] = 0u;

        while (stackSize > 0u)
        {
            const uint nodeIndex = stack[--stackSize];
            const GpuBvhNode node = g_BvhNodes[nodeIndex];
            if (!NodeOverlapsQuery(node, queryMin, queryMax))
            {
                continue;
            }

            if (node.left < 0 && node.right < 0)
            {
                const uint otherBodyId = node.primitiveIdx;
                if (otherBodyId > bodyId)
                {
                    ++pairCount;
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

    // TODO: use a static BVH for static bodies
    for (uint otherBodyId = 0u; otherBodyId < rigidBodyCount; ++otherBodyId)
    {
        if ((g_BodyMeta[otherBodyId].flags & kBodyFlagDynamic) != 0u)
        {
            continue;
        }

        const GpuBodyAabb otherAabb = g_BodyAabbs[otherBodyId];
        if (AabbOverlaps(queryMin, queryMax, otherAabb.minBounds.xyz, otherAabb.maxBounds.xyz))
        {
            ++pairCount;
        }
    }

    g_PairCounts[activeIndex] = pairCount;
}
