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

StructuredBuffer<GpuMortonCodeElement> g_MortonCodesIn;
RWStructuredBuffer<uint> g_RadixBitFlags;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= activeMovingCount)
    {
        return;
    }

    g_RadixBitFlags[index] = (g_MortonCodesIn[index].mortonCode >> iterationIndex) & 1u;
}
