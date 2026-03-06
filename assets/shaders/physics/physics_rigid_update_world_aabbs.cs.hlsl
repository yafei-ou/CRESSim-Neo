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

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyScales;
StructuredBuffer<uint> g_RigidBodyColliderShapeTypes;
StructuredBuffer<float4> g_RigidBodyColliderParams;

RWStructuredBuffer<GpuBodyAabb> g_BodyAabbs;
RWStructuredBuffer<GpuBodyMeta> g_BodyMeta;
RWStructuredBuffer<uint> g_ActiveBodyFlags;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    const float4 positionInvMass = g_PredictedRigidBodyPositionsInvMass[bodyIndex];
    const float4 orientation = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyIndex]);
    const float4 scale = g_RigidBodyScales[bodyIndex];
    const uint shapeType = g_RigidBodyColliderShapeTypes[bodyIndex];
    const float4 colliderParams = g_RigidBodyColliderParams[bodyIndex];

    float3 aabbMin = 0.0;
    float3 aabbMax = 0.0;
    ComputeBodyAabb(shapeType, positionInvMass.xyz, orientation, colliderParams, scale, aabbMin,
                    aabbMax);
    aabbMin -= float3(kBroadPhaseMargin, kBroadPhaseMargin, kBroadPhaseMargin);
    aabbMax += float3(kBroadPhaseMargin, kBroadPhaseMargin, kBroadPhaseMargin);

    GpuBodyAabb bodyAabb;
    bodyAabb.minBounds = float4(aabbMin, 0.0);
    bodyAabb.maxBounds = float4(aabbMax, 0.0);
    g_BodyAabbs[bodyIndex] = bodyAabb;

    const bool isDynamic = positionInvMass.w > 0.0;
    GpuBodyMeta meta;
    meta.bodyId = bodyIndex;
    meta.flags = isDynamic ? kBodyFlagDynamic : 0u;
    meta.activeIndex = kInvalidIndex;
    meta.reserved = 0u;
    g_BodyMeta[bodyIndex] = meta;
    g_ActiveBodyFlags[bodyIndex] = isDynamic ? 1u : 0u;
}
