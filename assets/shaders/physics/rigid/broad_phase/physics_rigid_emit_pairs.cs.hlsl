#include "physics_rigid_dispatch_constants.hlsli"
#include "physics_rigid_types.hlsli"
#include "physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_BroadPhaseBodyIndices);
CRESSIM_STRUCTURED_BUFFER(GpuBodyAabb, g_BodyAabbs);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_BvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuBvhNode, g_StaticBvhNodes);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);
CRESSIM_STRUCTURED_BUFFER(uint, g_JointCollisionSuppressionOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_JointCollisionSuppressionNeighbors);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsSphereSphere);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsSphereBox);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsSphereCapsule);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsBoxBox);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsBoxCapsule);
CRESSIM_STRUCTURED_BUFFER(uint, g_PairOffsetsCapsuleCapsule);
CRESSIM_STRUCTURED_BUFFER(GpuRigidPairRange, g_RigidPairRanges);
CRESSIM_RW_STRUCTURED_BUFFER(GpuCandidatePair, g_CandidatePairs);

void EmitCanonicalPair(uint colliderA, uint colliderB, uint shapeTypeA, uint shapeTypeB,
                       inout uint writeIndices[kRigidPairTypeCount])
{
    uint canonicalColliderA = 0u;
    uint canonicalColliderB = 0u;
    uint pairType = 0u;
    CanonicalizeRigidPair(colliderA, colliderB, shapeTypeA, shapeTypeB, canonicalColliderA,
                          canonicalColliderB, pairType);

    const uint writeIndex = writeIndices[pairType];
    writeIndices[pairType] = writeIndex + 1u;
    if (writeIndex >= candidatePairCapacity)
    {
        return;
    }

    GpuCandidatePair pair;
    pair.colliderA = canonicalColliderA;
    pair.colliderB = canonicalColliderB;
    pair.reserved0 = 0u;
    pair.reserved1 = 0u;
    CRESSIM_SB_STORE(g_CandidatePairs, writeIndex, pair);
}

bool IsJointCollisionSuppressed(uint bodyA, uint bodyB)
{
    const uint begin = CRESSIM_SB_LOAD(g_JointCollisionSuppressionOffsets, bodyA);
    const uint end = CRESSIM_SB_LOAD(g_JointCollisionSuppressionOffsets, bodyA + 1u);
    for (uint i = begin; i < end; ++i)
    {
        if (CRESSIM_SB_LOAD(g_JointCollisionSuppressionNeighbors, i) == bodyB)
        {
            return true;
        }
    }
    return false;
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

    uint writeIndices[kRigidPairTypeCount];
    writeIndices[0] = CRESSIM_SB_LOAD(g_RigidPairRanges, 0).start + CRESSIM_SB_LOAD(g_PairOffsetsSphereSphere, activeIndex);
    writeIndices[1] = CRESSIM_SB_LOAD(g_RigidPairRanges, 1).start + CRESSIM_SB_LOAD(g_PairOffsetsSphereBox, activeIndex);
    writeIndices[2] = CRESSIM_SB_LOAD(g_RigidPairRanges, 2).start + CRESSIM_SB_LOAD(g_PairOffsetsSphereCapsule, activeIndex);
    writeIndices[3] = CRESSIM_SB_LOAD(g_RigidPairRanges, 3).start + CRESSIM_SB_LOAD(g_PairOffsetsBoxBox, activeIndex);
    writeIndices[4] = CRESSIM_SB_LOAD(g_RigidPairRanges, 4).start + CRESSIM_SB_LOAD(g_PairOffsetsBoxCapsule, activeIndex);
    writeIndices[5] = CRESSIM_SB_LOAD(g_RigidPairRanges, 5).start + CRESSIM_SB_LOAD(g_PairOffsetsCapsuleCapsule, activeIndex);

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
                                                otherCollider.collisionMask) &&
                        !IsJointCollisionSuppressed(ownerBodyA, otherOwnerBody))
                    {
                        EmitCanonicalPair(colliderId, otherColliderId, shapeTypeA,
                                          otherCollider.shapeType, writeIndices);
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
                                                otherCollider.collisionMask) &&
                        !IsJointCollisionSuppressed(ownerBodyA, otherOwnerBody))
                    {
                        EmitCanonicalPair(colliderId, otherColliderId, shapeTypeA,
                                          otherCollider.shapeType, writeIndices);
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
}
