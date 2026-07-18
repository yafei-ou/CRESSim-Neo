#include "physics_rigid_dispatch_constants.hlsli"
#include "physics_rigid_types.hlsli"
#include "physics_rigid_broad_phase_types.hlsli"
#include "physics_shape_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ColliderOwnerRigidBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_ColliderShapeTypes);
CRESSIM_STRUCTURED_BUFFER(GpuColliderGeometryData, g_ColliderGeometryData);
CRESSIM_STRUCTURED_BUFFER(uint, g_ColliderEnabledFlags);

CRESSIM_RW_STRUCTURED_BUFFER(GpuBodyAabb, g_BodyAabbs);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBodyMeta, g_BodyMeta);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_ActiveBodyFlags);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_StaticBodyFlags);

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
    meta.bodyType = kRigidBodyTypeStatic;
    meta.activeIndex = kInvalidIndex;
    meta.reserved0 = 0u;

    const uint ownerBodyIndex = CRESSIM_SB_LOAD(g_ColliderOwnerRigidBodyIndices, colliderIndex);
    if (ownerBodyIndex >= rigidBodyCount ||
        CRESSIM_SB_LOAD(g_ColliderEnabledFlags, colliderIndex) == 0u)
    {
        CRESSIM_SB_STORE(g_BodyAabbs, colliderIndex, bodyAabb);
        CRESSIM_SB_STORE(g_BodyMeta, colliderIndex, meta);
        CRESSIM_SB_STORE(g_ActiveBodyFlags, colliderIndex, 0u);
        CRESSIM_SB_STORE(g_StaticBodyFlags, colliderIndex, 0u);
        return;
    }

    const float4 bodyPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, ownerBodyIndex);
    const float4 bodyOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, ownerBodyIndex));
    const float4 previousBodyPositionInvMass =
        CRESSIM_SB_LOAD(g_PreviousRigidBodyPositionsInvMass, ownerBodyIndex);
    const float4 previousBodyOrientation = QuaternionNormalize(
        CRESSIM_SB_LOAD(g_PreviousRigidBodyOrientations, ownerBodyIndex));
    const float4 scale = CRESSIM_SB_LOAD(g_RigidBodyScales, ownerBodyIndex);
    const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, ownerBodyIndex);
    const uint shapeType = CRESSIM_SB_LOAD(g_ColliderShapeTypes, colliderIndex);
    const GpuColliderGeometryData colliderGeometry =
        CRESSIM_SB_LOAD(g_ColliderGeometryData, colliderIndex);
    const float4 colliderParams = colliderGeometry.shapeParams;
    const float3 localPosition = colliderGeometry.localPosition.xyz * scale.xyz;
    const float4 localOrientation =
        QuaternionNormalize(colliderGeometry.localOrientation);
    const float3 colliderPosition =
        ComposeColliderWorldPosition(bodyPositionInvMass.xyz, bodyOrientation, localPosition);
    const float4 colliderOrientation =
        ComposeColliderWorldOrientation(bodyOrientation, localOrientation);
    const float3 previousColliderPosition = ComposeColliderWorldPosition(
        previousBodyPositionInvMass.xyz, previousBodyOrientation, localPosition);
    const float4 previousColliderOrientation =
        ComposeColliderWorldOrientation(previousBodyOrientation, localOrientation);

    float3 aabbMin = 0.0;
    float3 aabbMax = 0.0;
    float3 previousAabbMin = 0.0;
    float3 previousAabbMax = 0.0;
    ComputeBodyAabb(shapeType, colliderPosition, colliderOrientation, colliderParams, scale, aabbMin,
                    aabbMax);
    ComputeBodyAabb(shapeType, previousColliderPosition, previousColliderOrientation,
                    colliderParams, scale, previousAabbMin, previousAabbMax);
    aabbMin = min(aabbMin, previousAabbMin);
    aabbMax = max(aabbMax, previousAabbMax);
    aabbMin -= float3(kBroadPhaseMargin, kBroadPhaseMargin, kBroadPhaseMargin);
    aabbMax += float3(kBroadPhaseMargin, kBroadPhaseMargin, kBroadPhaseMargin);

    bodyAabb.minBounds = float4(aabbMin, 0.0);
    bodyAabb.maxBounds = float4(aabbMax, 0.0);
    CRESSIM_SB_STORE(g_BodyAabbs, colliderIndex, bodyAabb);

    meta.bodyType = bodyType;
    CRESSIM_SB_STORE(g_BodyMeta, colliderIndex, meta);
    CRESSIM_SB_STORE(
        g_ActiveBodyFlags, colliderIndex,
        bodyType == kRigidBodyTypeKinematic || bodyType == kRigidBodyTypeDynamic ? 1u : 0u);
    CRESSIM_SB_STORE(g_StaticBodyFlags, colliderIndex,
                     bodyType == kRigidBodyTypeStatic ? 1u : 0u);
}
