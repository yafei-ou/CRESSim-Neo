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

StructuredBuffer<uint> g_ScannedBlockOffsets;
RWStructuredBuffer<uint> g_ScanOutput;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID,
                                 uint3 groupID : SV_GroupID)
{
    const uint index = dispatchThreadID.x;
    if (index >= candidatePairCapacity)
    {
        return;
    }

    g_ScanOutput[index] += g_ScannedBlockOffsets[groupID.x];
}
