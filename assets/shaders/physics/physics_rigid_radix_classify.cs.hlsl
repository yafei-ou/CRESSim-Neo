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
RWStructuredBuffer<uint> g_RadixBitFlags;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= activeDynamicCount)
    {
        return;
    }

    g_RadixBitFlags[index] = (g_MortonCodesIn[index].mortonCode >> iterationIndex) & 1u;
}
