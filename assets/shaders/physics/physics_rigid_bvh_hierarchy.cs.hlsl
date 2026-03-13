#include "physics/include/physics_rigid_broad_phase_build_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<GpuMortonCodeElement> g_SortedMortonCodes;
StructuredBuffer<GpuBroadPhaseElement> g_BroadPhaseElements;
RWStructuredBuffer<GpuBvhNode> g_BvhNodes;
RWStructuredBuffer<GpuBvhConstructionInfo> g_BvhConstructionInfos;

int Delta(int i, uint codeI, int j)
{
    if (j < 0 || j >= int(elementCount))
    {
        return -1;
    }

    const uint codeJ = g_SortedMortonCodes[j].mortonCode;
    if (codeI == codeJ)
    {
        const uint elementIdxI = g_SortedMortonCodes[i].elementIdx;
        const uint elementIdxJ = g_SortedMortonCodes[j].elementIdx;
        const uint xorValue = elementIdxI ^ elementIdxJ;
        return (xorValue == 0u) ? 64 : (63 - firstbithigh(xorValue));
    }

    return 31 - firstbithigh(codeI ^ codeJ);
}

void DetermineRange(int idx, out int lower, out int upper)
{
    const uint code = g_SortedMortonCodes[idx].mortonCode;
    const int deltaL = Delta(idx, code, idx - 1);
    const int deltaR = Delta(idx, code, idx + 1);
    const int direction = (deltaR >= deltaL) ? 1 : -1;
    const int deltaMin = min(deltaL, deltaR);

    int lengthMax = 2;
    while (Delta(idx, code, idx + lengthMax * direction) > deltaMin)
    {
        lengthMax <<= 1;
    }

    int length = 0;
    for (int stride = lengthMax >> 1; stride > 0; stride >>= 1)
    {
        if (Delta(idx, code, idx + (length + stride) * direction) > deltaMin)
        {
            length += stride;
        }
    }

    const int other = idx + length * direction;
    lower = min(idx, other);
    upper = max(idx, other);
}

int FindSplit(int first, int last)
{
    const uint firstCode = g_SortedMortonCodes[first].mortonCode;
    const int commonPrefix = Delta(first, firstCode, last);

    int split = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int newSplit = split + stride;
        if (newSplit < last)
        {
            const int splitPrefix = Delta(first, firstCode, newSplit);
            if (splitPrefix > commonPrefix)
            {
                split = newSplit;
            }
        }
    } while (stride > 1);

    return split;
}

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint globalId = dispatchThreadID.x;
    if (elementCount == 0u)
    {
        return;
    }

    if (elementCount == 1u)
    {
        // root as a leaf node
        if (globalId == 0u)
        {
          const GpuBroadPhaseElement element =
              g_BroadPhaseElements[g_SortedMortonCodes[0].elementIdx];
          GpuBvhNode node;
          node.left = -1;
          node.right = -1;
          node.primitiveIdx = element.primitiveIdx;
          node.aabbMinX = element.aabbMinX;
          node.aabbMinY = element.aabbMinY;
          node.aabbMinZ = element.aabbMinZ;
          node.aabbMaxX = element.aabbMaxX;
          node.aabbMaxY = element.aabbMaxY;
          node.aabbMaxZ = element.aabbMaxZ;
          node.reserved = 0.0;
          g_BvhNodes[0] = node;

          GpuBvhConstructionInfo rootInfo;
          rootInfo.parent = 0u;
          rootInfo.visitationCount = 0;
          g_BvhConstructionInfos[0] = rootInfo;
        }
        return;
    }

    const int leafOffset = int(elementCount) - 1;
    if (globalId < elementCount)
    {
        const GpuBroadPhaseElement element =
            g_BroadPhaseElements[g_SortedMortonCodes[globalId].elementIdx];
        GpuBvhNode node;
        node.left = -1;
        node.right = -1;
        node.primitiveIdx = element.primitiveIdx;
        node.aabbMinX = element.aabbMinX;
        node.aabbMinY = element.aabbMinY;
        node.aabbMinZ = element.aabbMinZ;
        node.aabbMaxX = element.aabbMaxX;
        node.aabbMaxY = element.aabbMaxY;
        node.aabbMaxZ = element.aabbMaxZ;
        node.reserved = 0.0;
        g_BvhNodes[leafOffset + globalId] = node;
    }

    if (globalId < elementCount - 1u)
    {
        int first = 0;
        int last = 0;
        DetermineRange(int(globalId), first, last);
        const int split = FindSplit(first, last);

        const int childA = (split == first) ? (leafOffset + split) : split;
        const int childB = (split + 1 == last) ? (leafOffset + split + 1) : (split + 1);

        GpuBvhNode node;
        node.left = childA;
        node.right = childB;
        node.primitiveIdx = 0u;
        node.aabbMinX = 0.0;
        node.aabbMinY = 0.0;
        node.aabbMinZ = 0.0;
        node.aabbMaxX = 0.0;
        node.aabbMaxY = 0.0;
        node.aabbMaxZ = 0.0;
        node.reserved = 0.0;
        g_BvhNodes[globalId] = node;

        GpuBvhConstructionInfo childInfoA;
        childInfoA.parent = globalId;
        childInfoA.visitationCount = 0;
        g_BvhConstructionInfos[childA] = childInfoA;

        GpuBvhConstructionInfo childInfoB;
        childInfoB.parent = globalId;
        childInfoB.visitationCount = 0;
        g_BvhConstructionInfos[childB] = childInfoB;
    }

    if (globalId == 0u)
    {
        GpuBvhConstructionInfo rootInfo;
        rootInfo.parent = 0u;
        rootInfo.visitationCount = 0;
        g_BvhConstructionInfos[0] = rootInfo;
    }
}
