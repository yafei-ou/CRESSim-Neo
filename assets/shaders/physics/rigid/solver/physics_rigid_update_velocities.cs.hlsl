#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    const float4 previousPositionInvMass = CRESSIM_SB_LOAD(g_PreviousRigidBodyPositionsInvMass, bodyIndex);
    const float4 previousOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PreviousRigidBodyOrientations, bodyIndex));
    const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyIndex);
    float4 predictedPositionInvMass = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyIndex);
    float4 predictedOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyIndex));

    if (bodyType == kRigidBodyTypeStatic)
    {
        predictedOrientation = QuaternionNormalize(predictedOrientation);
        CRESSIM_SB_STORE(g_PredictedRigidBodyPositionsInvMass, bodyIndex, predictedPositionInvMass);
        CRESSIM_SB_STORE(g_PredictedRigidBodyOrientations, bodyIndex, predictedOrientation);
        CRESSIM_SB_STORE(g_PredictedRigidBodyLinearVelocities, bodyIndex, 0.0);
        CRESSIM_SB_STORE(g_PredictedRigidBodyAngularVelocities, bodyIndex, 0.0);
        return;
    }

    const float3 linearVelocity =
        (predictedPositionInvMass.xyz - previousPositionInvMass.xyz) / max(dt, kEpsilon);
    const float3 angularVelocity =
        AngularVelocityFromQuaternionDelta(previousOrientation, predictedOrientation,
                                           max(dt, kEpsilon));

    CRESSIM_SB_STORE(g_PredictedRigidBodyPositionsInvMass, bodyIndex, predictedPositionInvMass);
    CRESSIM_SB_STORE(g_PredictedRigidBodyOrientations, bodyIndex, predictedOrientation);
    CRESSIM_SB_STORE(g_PredictedRigidBodyLinearVelocities, bodyIndex, float4(linearVelocity, 0.0));
    CRESSIM_SB_STORE(g_PredictedRigidBodyAngularVelocities, bodyIndex, float4(angularVelocity, 0.0));
}
