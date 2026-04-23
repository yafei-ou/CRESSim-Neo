#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_BroadPhaseBodyIndices);
CRESSIM_STRUCTURED_BUFFER(GpuBodyAabb, g_BodyAabbs);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_BvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_StaticBvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PairCountsSphereSphere);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PairCountsSphereBox);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PairCountsSphereCapsule);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PairCountsBoxBox);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PairCountsBoxCapsule);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_PairCountsCapsuleCapsule);

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

    const uint colliderId = CRESSIM_SB_LOAD(g_BroadPhaseBodyIndices, activeIndex);
    const GpuColliderBroadPhaseData colliderA = CRESSIM_SB_LOAD(g_ColliderBroadPhaseData, colliderId);
    const uint ownerBodyA = colliderA.ownerBody;
    const uint environmentA = colliderA.environmentIndex;
    const uint shapeTypeA = colliderA.shapeType;
    const uint layerA = colliderA.collisionLayer;
    const uint maskA = colliderA.collisionMask;
    const GpuBodyAabb bodyAabb = CRESSIM_SB_LOAD(g_BodyAabbs, colliderId);
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
            const GpuBvhNode node = CRESSIM_SB_LOAD(g_BvhNodes, nodeIndex);
            if (!NodeAabbOverlapsQuery(node, queryMin, queryMax))
            {
                continue;
            }

            if (node.left < 0 && node.right < 0)
            {
                const uint otherColliderId = node.primitiveIdx;
                const GpuColliderBroadPhaseData otherCollider =
                    CRESSIM_SB_LOAD(g_ColliderBroadPhaseData, otherColliderId);
                const uint otherOwnerBody = otherCollider.ownerBody;
                if (otherColliderId > colliderId && otherOwnerBody != ownerBodyA)
                {
                    if (ShouldBroadPhaseCollide(environmentA, otherCollider.environmentIndex, layerA,
                                                maskA, otherCollider.collisionLayer,
                                                otherCollider.collisionMask))
                    {
                        IncrementTypedCount(ComputeRigidPairType(shapeTypeA,
                                                                 otherCollider.shapeType),
                                            typedCounts);
                    }
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
            const GpuBvhNode node = CRESSIM_SB_LOAD(g_StaticBvhNodes, nodeIndex);
            if (!NodeAabbOverlapsQuery(node, queryMin, queryMax))
            {
                continue;
            }

            if (node.left < 0 && node.right < 0)
            {
                const uint otherColliderId = node.primitiveIdx;
                const GpuColliderBroadPhaseData otherCollider =
                    CRESSIM_SB_LOAD(g_ColliderBroadPhaseData, otherColliderId);
                const uint otherOwnerBody = otherCollider.ownerBody;
                if (otherOwnerBody != ownerBodyA)
                {
                    if (ShouldBroadPhaseCollide(environmentA, otherCollider.environmentIndex, layerA,
                                                maskA, otherCollider.collisionLayer,
                                                otherCollider.collisionMask))
                    {
                        IncrementTypedCount(ComputeRigidPairType(shapeTypeA,
                                                                 otherCollider.shapeType),
                                            typedCounts);
                    }
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

    CRESSIM_SB_STORE(g_PairCountsSphereSphere, activeIndex, typedCounts[0]);
    CRESSIM_SB_STORE(g_PairCountsSphereBox, activeIndex, typedCounts[1]);
    CRESSIM_SB_STORE(g_PairCountsSphereCapsule, activeIndex, typedCounts[2]);
    CRESSIM_SB_STORE(g_PairCountsBoxBox, activeIndex, typedCounts[3]);
    CRESSIM_SB_STORE(g_PairCountsBoxCapsule, activeIndex, typedCounts[4]);
    CRESSIM_SB_STORE(g_PairCountsCapsuleCapsule, activeIndex, typedCounts[5]);
}
