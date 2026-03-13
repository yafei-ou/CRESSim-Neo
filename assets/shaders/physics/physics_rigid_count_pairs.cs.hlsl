#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_ActiveBodyIndices;
StructuredBuffer<GpuBodyAabb> g_BodyAabbs;
StructuredBuffer<GpuBvhNode> g_BvhNodes;
StructuredBuffer<GpuBvhNode> g_StaticBvhNodes;
StructuredBuffer<uint> g_RigidBodyColliderShapeTypes;
RWStructuredBuffer<uint> g_PairCountsSphereSphere;
RWStructuredBuffer<uint> g_PairCountsSphereBox;
RWStructuredBuffer<uint> g_PairCountsSphereCapsule;
RWStructuredBuffer<uint> g_PairCountsBoxBox;
RWStructuredBuffer<uint> g_PairCountsBoxCapsule;
RWStructuredBuffer<uint> g_PairCountsCapsuleCapsule;

bool NodeOverlapsQuery(GpuBvhNode node, float3 queryMin, float3 queryMax)
{
    return AabbOverlaps(queryMin, queryMax, float3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
                        float3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ));
}

void IncrementTypedCount(uint pairType, inout uint counts[kRigidPairTypeCount])
{
    if (pairType < kRigidPairTypeCount)
    {
        ++counts[pairType];
    }
}

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint activeIndex = dispatchThreadID.x;
    if (activeIndex >= activeMovingCount)
    {
        return;
    }

    const uint bodyId = g_ActiveBodyIndices[activeIndex];
    const uint shapeTypeA = g_RigidBodyColliderShapeTypes[bodyId];
    const GpuBodyAabb bodyAabb = g_BodyAabbs[bodyId];
    const float3 queryMin = bodyAabb.minBounds.xyz;
    const float3 queryMax = bodyAabb.maxBounds.xyz;

    uint typedCounts[kRigidPairTypeCount];
    [unroll] for (uint i = 0u; i < kRigidPairTypeCount; ++i)
    {
        typedCounts[i] = 0u;
    }

    if (activeMovingCount > 1u)
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
                    IncrementTypedCount(
                        ComputeRigidPairType(shapeTypeA, g_RigidBodyColliderShapeTypes[otherBodyId]),
                        typedCounts);
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

    if (staticBodyCount > 0u)
    {
        uint stack[128];
        uint stackSize = 0u;
        stack[stackSize++] = 0u;

        while (stackSize > 0u)
        {
            const uint nodeIndex = stack[--stackSize];
            const GpuBvhNode node = g_StaticBvhNodes[nodeIndex];
            if (!NodeOverlapsQuery(node, queryMin, queryMax))
            {
                continue;
            }

            if (node.left < 0 && node.right < 0)
            {
                const uint otherBodyId = node.primitiveIdx;
                IncrementTypedCount(
                    ComputeRigidPairType(shapeTypeA, g_RigidBodyColliderShapeTypes[otherBodyId]),
                    typedCounts);
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

    g_PairCountsSphereSphere[activeIndex] = typedCounts[0];
    g_PairCountsSphereBox[activeIndex] = typedCounts[1];
    g_PairCountsSphereCapsule[activeIndex] = typedCounts[2];
    g_PairCountsBoxBox[activeIndex] = typedCounts[3];
    g_PairCountsBoxCapsule[activeIndex] = typedCounts[4];
    g_PairCountsCapsuleCapsule[activeIndex] = typedCounts[5];
}
