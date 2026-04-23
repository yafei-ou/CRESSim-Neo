#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "../../../include/physics/collision/physics_shape_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ColliderOwnerRigidBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_ColliderShapeTypes);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderShapeParams);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderLocalPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderLocalOrientations);
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

    const float4 bodyPositionInvMass = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, ownerBodyIndex);
    const float4 bodyOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, ownerBodyIndex));
    const float4 scale = CRESSIM_SB_LOAD(g_RigidBodyScales, ownerBodyIndex);
    const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, ownerBodyIndex);
    const uint shapeType = CRESSIM_SB_LOAD(g_ColliderShapeTypes, colliderIndex);
    const float4 colliderParams = CRESSIM_SB_LOAD(g_ColliderShapeParams, colliderIndex);
    const float3 localPosition = CRESSIM_SB_REF(g_ColliderLocalPositions, colliderIndex).xyz * scale.xyz;
    const float4 localOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_ColliderLocalOrientations, colliderIndex));
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
    CRESSIM_SB_STORE(g_BodyAabbs, colliderIndex, bodyAabb);

    meta.bodyType = bodyType;
    CRESSIM_SB_STORE(g_BodyMeta, colliderIndex, meta);
    CRESSIM_SB_STORE(
        g_ActiveBodyFlags, colliderIndex,
        bodyType == kRigidBodyTypeKinematic || bodyType == kRigidBodyTypeDynamic ? 1u : 0u);
    CRESSIM_SB_STORE(g_StaticBodyFlags, colliderIndex,
                     bodyType == kRigidBodyTypeStatic ? 1u : 0u);
}
