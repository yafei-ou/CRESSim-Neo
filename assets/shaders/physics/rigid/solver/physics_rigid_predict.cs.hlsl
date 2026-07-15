#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyKinematicTargetOrientations);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyKinematicTargetFlags);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= rigidBodyCount)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_RigidBodyPositionsInvMass, idx);
    const float4 orientation = QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyOrientations, idx));
    float4 linearVelocity = CRESSIM_SB_LOAD(g_RigidBodyLinearVelocities, idx);
    float4 angularVelocity = CRESSIM_SB_LOAD(g_RigidBodyAngularVelocities, idx);
    const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, idx);
    const float4 kinematicTargetPosition = CRESSIM_SB_LOAD(g_RigidBodyKinematicTargetPositions, idx);
    const float4 kinematicTargetOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_RigidBodyKinematicTargetOrientations, idx));
    const bool kinematicTargetEnabled =
        (CRESSIM_SB_LOAD(g_RigidBodyKinematicTargetFlags, idx) & kKinematicTargetEnabled) != 0u;

    CRESSIM_SB_STORE(g_PreviousRigidBodyPositionsInvMass, idx, positionInvMass);
    CRESSIM_SB_STORE(g_PreviousRigidBodyOrientations, idx, orientation);

    if (bodyType == kRigidBodyTypeStatic)
    {
        CRESSIM_SB_STORE(g_PredictedRigidBodyPositionsInvMass, idx, positionInvMass);
        CRESSIM_SB_STORE(g_PredictedRigidBodyOrientations, idx, orientation);
        CRESSIM_SB_STORE(g_PredictedRigidBodyLinearVelocities, idx, float4(0.0, 0.0, 0.0, 0.0));
        CRESSIM_SB_STORE(g_PredictedRigidBodyAngularVelocities, idx, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    if (bodyType == kRigidBodyTypeKinematic)
    {
        const float3 predictedPosition =
            kinematicTargetEnabled ? kinematicTargetPosition.xyz : positionInvMass.xyz;
        const float4 predictedOrientation =
            kinematicTargetEnabled ? kinematicTargetOrientation : orientation;
        linearVelocity.xyz = (predictedPosition - positionInvMass.xyz) / max(dt, kEpsilon);
        angularVelocity.xyz =
            AngularVelocityFromQuaternionDelta(orientation, predictedOrientation, max(dt, kEpsilon));

        CRESSIM_SB_STORE(g_PredictedRigidBodyPositionsInvMass, idx,
                         float4(predictedPosition, positionInvMass.w));
        CRESSIM_SB_STORE(g_PredictedRigidBodyOrientations, idx, predictedOrientation);
        CRESSIM_SB_STORE(g_PredictedRigidBodyLinearVelocities, idx, linearVelocity);
        CRESSIM_SB_STORE(g_PredictedRigidBodyAngularVelocities, idx, angularVelocity);
        return;
    }

    linearVelocity.xyz += gravity.xyz * dt;
    const float3 predictedPosition = positionInvMass.xyz + linearVelocity.xyz * dt;
    const float4 predictedOrientation =
        IntegrateOrientation(orientation, angularVelocity.xyz, dt);

    CRESSIM_SB_STORE(g_PredictedRigidBodyPositionsInvMass, idx,
                     float4(predictedPosition, positionInvMass.w));
    CRESSIM_SB_STORE(g_PredictedRigidBodyOrientations, idx, predictedOrientation);
    CRESSIM_SB_STORE(g_PredictedRigidBodyLinearVelocities, idx, linearVelocity);
    CRESSIM_SB_STORE(g_PredictedRigidBodyAngularVelocities, idx, angularVelocity);
}
