#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyScales;
StructuredBuffer<uint> g_RigidBodyTypes;
StructuredBuffer<uint> g_ColliderOwnerRigidBodyIndices;
StructuredBuffer<uint> g_ColliderShapeTypes;
StructuredBuffer<float4> g_ColliderShapeParams;
StructuredBuffer<float4> g_ColliderLocalPositions;
StructuredBuffer<float4> g_ColliderLocalOrientations;
StructuredBuffer<uint> g_ColliderEnabledFlags;

RWStructuredBuffer<GpuBodyAabb> g_BodyAabbs;
RWStructuredBuffer<GpuBodyMeta> g_BodyMeta;
RWStructuredBuffer<uint> g_ActiveBodyFlags;
RWStructuredBuffer<uint> g_StaticBodyFlags;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint colliderIndex = dispatchThreadID.x;
    if (colliderIndex >= colliderCount)
    {
        return;
    }

    GpuBodyAabb bodyAabb;
    bodyAabb.minBounds = 0.0;
    bodyAabb.maxBounds = 0.0;

    GpuBodyMeta meta;
    meta.bodyId = colliderIndex;
    meta.flags = 0u;
    meta.activeIndex = kInvalidIndex;
    meta.reserved = 0u;

    const uint ownerBodyIndex = g_ColliderOwnerRigidBodyIndices[colliderIndex];
    if (ownerBodyIndex >= rigidBodyCount || g_ColliderEnabledFlags[colliderIndex] == 0u)
    {
        g_BodyAabbs[colliderIndex] = bodyAabb;
        g_BodyMeta[colliderIndex] = meta;
        g_ActiveBodyFlags[colliderIndex] = 0u;
        g_StaticBodyFlags[colliderIndex] = 0u;
        return;
    }

    const float4 bodyPositionInvMass = g_PredictedRigidBodyPositionsInvMass[ownerBodyIndex];
    const float4 bodyOrientation =
        QuaternionNormalize(g_PredictedRigidBodyOrientations[ownerBodyIndex]);
    const float4 scale = g_RigidBodyScales[ownerBodyIndex];
    const uint bodyType = g_RigidBodyTypes[ownerBodyIndex];
    const uint shapeType = g_ColliderShapeTypes[colliderIndex];
    const float4 colliderParams = g_ColliderShapeParams[colliderIndex];
    const float3 localPosition = g_ColliderLocalPositions[colliderIndex].xyz * scale.xyz;
    const float4 localOrientation =
        QuaternionNormalize(g_ColliderLocalOrientations[colliderIndex]);
    const float3 colliderPosition =
        ComposeColliderWorldPosition(bodyPositionInvMass.xyz, bodyOrientation, localPosition);
    const float4 colliderOrientation =
        ComposeColliderWorldOrientation(bodyOrientation, localOrientation);

    float3 aabbMin = 0.0;
    float3 aabbMax = 0.0;
    ComputeBodyAabb(shapeType, colliderPosition, colliderOrientation, colliderParams, scale, aabbMin,
                    aabbMax);
    aabbMin -= float3(kBroadPhaseMargin, kBroadPhaseMargin, kBroadPhaseMargin);
    aabbMax += float3(kBroadPhaseMargin, kBroadPhaseMargin, kBroadPhaseMargin);

    bodyAabb.minBounds = float4(aabbMin, 0.0);
    bodyAabb.maxBounds = float4(aabbMax, 0.0);
    g_BodyAabbs[colliderIndex] = bodyAabb;

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
    g_BodyMeta[colliderIndex] = meta;
    g_ActiveBodyFlags[colliderIndex] = (meta.flags & kBodyFlagMoving) != 0u ? 1u : 0u;
    g_StaticBodyFlags[colliderIndex] = (meta.flags & kBodyFlagStatic) != 0u ? 1u : 0u;
}
