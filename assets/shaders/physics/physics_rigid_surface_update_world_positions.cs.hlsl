#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidSurfaceParticleLocalPositions);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleOwningRigidBodyIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleOwningColliderIndices);

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderLocalPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderLocalOrientations);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidSurfaceParticleWorldPositions);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= rigidSurfaceParticleCount)
    {
        return;
    }

    const uint bodyIndex = CRESSIM_SB_LOAD(g_RigidSurfaceParticleOwningRigidBodyIndices, idx);
    const uint colliderIndex = CRESSIM_SB_LOAD(g_RigidSurfaceParticleOwningColliderIndices, idx);

    const float3 bodyPosition = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyIndex).xyz;
    const float4 bodyOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyIndex));
    const float3 colliderLocalPosition = CRESSIM_SB_LOAD(g_ColliderLocalPositions, colliderIndex).xyz;
    const float4 colliderLocalOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_ColliderLocalOrientations, colliderIndex));
    const float3 sampleLocalPosition = CRESSIM_SB_LOAD(g_RigidSurfaceParticleLocalPositions, idx).xyz;

    const float3 colliderWorldPosition =
        bodyPosition + QuaternionRotate(bodyOrientation, colliderLocalPosition);
    const float4 colliderWorldOrientation =
        QuaternionNormalize(QuaternionMul(bodyOrientation, colliderLocalOrientation));
    const float3 sampleWorldPosition =
        colliderWorldPosition + QuaternionRotate(colliderWorldOrientation, sampleLocalPosition);

    CRESSIM_SB_STORE(g_RigidSurfaceParticleWorldPositions, idx, float4(sampleWorldPosition, 0.0));
}
