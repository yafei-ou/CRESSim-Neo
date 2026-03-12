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
