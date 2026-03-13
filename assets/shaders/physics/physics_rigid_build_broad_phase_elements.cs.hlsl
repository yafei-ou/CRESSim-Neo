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

StructuredBuffer<uint> g_ActiveBodyIndices;
StructuredBuffer<GpuBodyAabb> g_BodyAabbs;
RWStructuredBuffer<GpuBroadPhaseElement> g_BroadPhaseElements;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint activeIndex = dispatchThreadID.x;
    if (activeIndex >= activeMovingCount)
    {
        return;
    }

    const uint bodyId = g_ActiveBodyIndices[activeIndex];
    const GpuBodyAabb bodyAabb = g_BodyAabbs[bodyId];

    GpuBroadPhaseElement element;
    element.primitiveIdx = bodyId;
    element.aabbMinX = bodyAabb.minBounds.x;
    element.aabbMinY = bodyAabb.minBounds.y;
    element.aabbMinZ = bodyAabb.minBounds.z;
    element.aabbMaxX = bodyAabb.maxBounds.x;
    element.aabbMaxY = bodyAabb.maxBounds.y;
    element.aabbMaxZ = bodyAabb.maxBounds.z;
    element.reserved = 0.0;
    g_BroadPhaseElements[activeIndex] = element;
}
