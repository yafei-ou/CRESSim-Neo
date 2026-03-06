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

StructuredBuffer<uint> g_RadixBitFlags;
StructuredBuffer<uint> g_RadixBitOffsets;
RWStructuredBuffer<uint> g_RadixMeta;

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    if (activeDynamicCount == 0u)
    {
        g_RadixMeta[0] = 0u;
        return;
    }

    const uint totalOnes =
        g_RadixBitOffsets[activeDynamicCount - 1u] + g_RadixBitFlags[activeDynamicCount - 1u];
    g_RadixMeta[0] = activeDynamicCount - totalOnes;
}
