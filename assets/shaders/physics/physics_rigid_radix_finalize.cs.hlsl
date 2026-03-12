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

StructuredBuffer<uint> g_RadixBitFlags;
StructuredBuffer<uint> g_RadixBitOffsets;
RWStructuredBuffer<uint> g_RadixMeta;

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    if (activeMovingCount == 0u)
    {
        g_RadixMeta[0] = 0u;
        return;
    }

    const uint totalOnes =
        g_RadixBitOffsets[activeMovingCount - 1u] + g_RadixBitFlags[activeMovingCount - 1u];
    g_RadixMeta[0] = activeMovingCount - totalOnes;
}
