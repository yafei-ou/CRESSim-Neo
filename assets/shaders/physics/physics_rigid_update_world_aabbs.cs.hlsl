#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyScales;
StructuredBuffer<uint> g_RigidBodyTypes;
StructuredBuffer<uint> g_RigidBodyColliderShapeTypes;
StructuredBuffer<float4> g_RigidBodyColliderParams;

RWStructuredBuffer<GpuBodyAabb> g_BodyAabbs;
RWStructuredBuffer<GpuBodyMeta> g_BodyMeta;
RWStructuredBuffer<uint> g_ActiveBodyFlags;
RWStructuredBuffer<uint> g_StaticBodyFlags;

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
    const uint bodyType = g_RigidBodyTypes[bodyIndex];
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

    GpuBodyMeta meta;
    meta.bodyId = bodyIndex;
    meta.flags = 0u;
    if (bodyType == 0u)
    {
        meta.flags |= kBodyFlagStatic;
    }
    else if (bodyType == 1u)
    {
        meta.flags |= kBodyFlagKinematic;
    }
    else
    {
        meta.flags |= kBodyFlagDynamic;
    }
    meta.activeIndex = kInvalidIndex;
    meta.reserved = 0u;
    g_BodyMeta[bodyIndex] = meta;
    g_ActiveBodyFlags[bodyIndex] = (meta.flags & kBodyFlagMoving) != 0u ? 1u : 0u;
    g_StaticBodyFlags[bodyIndex] = (meta.flags & kBodyFlagStatic) != 0u ? 1u : 0u;
}
