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

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    uint running = 0u;
    for (uint i = 0u; i < candidatePairCapacity; ++i)
    {
        g_ScanOutput[i] = running;
        running += g_ScanInput[i];
    }
}
