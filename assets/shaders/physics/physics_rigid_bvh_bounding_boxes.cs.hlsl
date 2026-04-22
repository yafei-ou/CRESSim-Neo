#include "include/physics/physics_rigid_broad_phase_build_constants.hlsli"
#include "include/physics/physics_rigid_common.hlsli"

CRESSIM_GLOBALLYCOHERENT_RW_STRUCTURED_BUFFER(GpuBvhNode, g_BvhNodes);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBvhConstructionInfo, g_BvhConstructionInfos);

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint globalId = dispatchThreadID.x;
    if (elementCount < 2u || globalId >= elementCount)
    {
        return;
    }

    const uint leafOffset = elementCount - 1u;
    uint nodeIndex = CRESSIM_SB_LOAD(g_BvhConstructionInfos, leafOffset + globalId).parent;
    while (true)
    {
        int previousVisits = 0;
        InterlockedAdd(CRESSIM_SB_REF(g_BvhConstructionInfos, nodeIndex).visitationCount, 1, previousVisits);
        if (previousVisits < 1)
        {
            return;
        }

        GpuBvhNode node = CRESSIM_SB_LOAD(g_BvhNodes, nodeIndex);
        const GpuBvhNode childA = CRESSIM_SB_LOAD(g_BvhNodes, node.left);
        const GpuBvhNode childB = CRESSIM_SB_LOAD(g_BvhNodes, node.right);

        const float3 minA = float3(childA.aabbMinX, childA.aabbMinY, childA.aabbMinZ);
        const float3 maxA = float3(childA.aabbMaxX, childA.aabbMaxY, childA.aabbMaxZ);
        const float3 minB = float3(childB.aabbMinX, childB.aabbMinY, childB.aabbMinZ);
        const float3 maxB = float3(childB.aabbMaxX, childB.aabbMaxY, childB.aabbMaxZ);
        const float3 unionMin = min(minA, minB);
        const float3 unionMax = max(maxA, maxB);

        node.aabbMinX = unionMin.x;
        node.aabbMinY = unionMin.y;
        node.aabbMinZ = unionMin.z;
        node.aabbMaxX = unionMax.x;
        node.aabbMaxY = unionMax.y;
        node.aabbMaxZ = unionMax.z;
        CRESSIM_SB_STORE(g_BvhNodes, nodeIndex, node);

        if (nodeIndex == 0u)
        {
            return;
        }
        nodeIndex = CRESSIM_SB_LOAD(g_BvhConstructionInfos, nodeIndex).parent;
    }
}
