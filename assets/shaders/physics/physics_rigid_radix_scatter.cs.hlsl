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

StructuredBuffer<GpuMortonCodeElement> g_MortonCodesIn;
StructuredBuffer<uint> g_RadixBitFlags;
StructuredBuffer<uint> g_RadixBitOffsets;
StructuredBuffer<uint> g_RadixMeta;
RWStructuredBuffer<GpuMortonCodeElement> g_MortonCodesOut;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= activeDynamicCount)
    {
        return;
    }

    const uint flag = g_RadixBitFlags[index];
    const uint offset = g_RadixBitOffsets[index];
    const uint totalZeros = g_RadixMeta[0];
    const uint dstIndex = (flag == 0u) ? (index - offset) : (totalZeros + offset);
    g_MortonCodesOut[dstIndex] = g_MortonCodesIn[index];
}
