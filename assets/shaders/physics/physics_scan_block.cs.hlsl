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

StructuredBuffer<uint> g_ScanInput;
RWStructuredBuffer<uint> g_ScanOutput;
RWStructuredBuffer<uint> g_BlockSums;

groupshared uint g_SharedScan[64];

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID,
                                 uint3 groupThreadID : SV_GroupThreadID,
                                 uint3 groupID : SV_GroupID)
{
    const uint index = dispatchThreadID.x;
    const uint localIndex = groupThreadID.x;
    const uint groupBase = groupID.x * 64u;
    const uint count = candidatePairCapacity;

    uint value = 0u;
    if (index < count)
    {
        value = g_ScanInput[index];
    }

    g_SharedScan[localIndex] = value;
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 1u; offset < 64u; offset <<= 1u)
    {
        uint addend = 0u;
        if (localIndex >= offset)
        {
            addend = g_SharedScan[localIndex - offset];
        }

        GroupMemoryBarrierWithGroupSync();
        g_SharedScan[localIndex] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    if (index < count)
    {
        g_ScanOutput[index] = g_SharedScan[localIndex] - value;
    }

    const uint remaining = (groupBase < count) ? (count - groupBase) : 0u;
    const uint activeCount = min(remaining, 64u);
    if (activeCount > 0u && localIndex == activeCount - 1u)
    {
        g_BlockSums[groupID.x] = g_SharedScan[localIndex];
    }
}
