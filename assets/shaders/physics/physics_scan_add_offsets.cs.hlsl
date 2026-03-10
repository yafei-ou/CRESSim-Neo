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
