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

RWStructuredBuffer<int4> g_RigidBodyTranslationCorrections;
RWStructuredBuffer<int4> g_RigidBodyRotationCorrections;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    g_RigidBodyTranslationCorrections[bodyIndex] = int4(0, 0, 0, 0);
    g_RigidBodyRotationCorrections[bodyIndex] = int4(0, 0, 0, 0);
}
