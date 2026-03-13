cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint activeMovingCount;
    uint staticBodyCount;
    uint candidatePairCount;
    uint candidatePairCapacity;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
    uint reserved0;
    uint reserved1;
};

#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_ActiveBodyIndices;
StructuredBuffer<GpuBodyAabb> g_BodyAabbs;
StructuredBuffer<GpuBvhNode> g_BvhNodes;
StructuredBuffer<GpuBvhNode> g_StaticBvhNodes;
StructuredBuffer<uint> g_RigidBodyColliderShapeTypes;
StructuredBuffer<uint> g_PairOffsetsSphereSphere;
StructuredBuffer<uint> g_PairOffsetsSphereBox;
StructuredBuffer<uint> g_PairOffsetsSphereCapsule;
StructuredBuffer<uint> g_PairOffsetsBoxBox;
StructuredBuffer<uint> g_PairOffsetsBoxCapsule;
StructuredBuffer<uint> g_PairOffsetsCapsuleCapsule;
StructuredBuffer<GpuRigidPairRange> g_RigidPairRanges;
RWStructuredBuffer<GpuCandidatePair> g_CandidatePairs;

bool NodeOverlapsQuery(GpuBvhNode node, float3 queryMin, float3 queryMax)
{
    return AabbOverlaps(queryMin, queryMax, float3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
                        float3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ));
}

void EmitCanonicalPair(uint bodyA, uint bodyB, uint shapeTypeA, uint shapeTypeB,
                       inout uint writeIndices[kRigidPairTypeCount])
{
    uint canonicalBodyA = 0u;
    uint canonicalBodyB = 0u;
    uint pairType = 0u;
    CanonicalizeRigidPair(bodyA, bodyB, shapeTypeA, shapeTypeB, canonicalBodyA, canonicalBodyB,
                          pairType);

    const uint writeIndex = writeIndices[pairType];
    writeIndices[pairType] = writeIndex + 1u;
    if (writeIndex >= candidatePairCapacity)
    {
        return;
    }

    GpuCandidatePair pair;
    pair.bodyA = canonicalBodyA;
    pair.bodyB = canonicalBodyB;
    pair.reserved0 = 0u;
    pair.reserved1 = 0u;
    g_CandidatePairs[writeIndex] = pair;
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

    uint writeIndices[kRigidPairTypeCount];
    writeIndices[0] = g_RigidPairRanges[0].start + g_PairOffsetsSphereSphere[activeIndex];
    writeIndices[1] = g_RigidPairRanges[1].start + g_PairOffsetsSphereBox[activeIndex];
    writeIndices[2] = g_RigidPairRanges[2].start + g_PairOffsetsSphereCapsule[activeIndex];
    writeIndices[3] = g_RigidPairRanges[3].start + g_PairOffsetsBoxBox[activeIndex];
    writeIndices[4] = g_RigidPairRanges[4].start + g_PairOffsetsBoxCapsule[activeIndex];
    writeIndices[5] = g_RigidPairRanges[5].start + g_PairOffsetsCapsuleCapsule[activeIndex];

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
                    EmitCanonicalPair(bodyId, otherBodyId, shapeTypeA,
                                      g_RigidBodyColliderShapeTypes[otherBodyId], writeIndices);
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
                EmitCanonicalPair(bodyId, otherBodyId, shapeTypeA,
                                  g_RigidBodyColliderShapeTypes[otherBodyId], writeIndices);
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
