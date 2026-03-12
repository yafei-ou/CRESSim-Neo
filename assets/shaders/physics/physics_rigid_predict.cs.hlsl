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

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_RigidBodyPositionsInvMass;
StructuredBuffer<float4> g_RigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyLinearVelocities;
StructuredBuffer<float4> g_RigidBodyAngularVelocities;
StructuredBuffer<uint> g_RigidBodyTypes;
StructuredBuffer<float4> g_RigidBodyKinematicTargetPositions;
StructuredBuffer<float4> g_RigidBodyKinematicTargetOrientations;
StructuredBuffer<uint> g_RigidBodyKinematicTargetFlags;

RWStructuredBuffer<float4> g_PreviousRigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_PreviousRigidBodyOrientations;
RWStructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_PredictedRigidBodyOrientations;
RWStructuredBuffer<float4> g_PredictedRigidBodyLinearVelocities;
RWStructuredBuffer<float4> g_PredictedRigidBodyAngularVelocities;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= rigidBodyCount)
    {
        return;
    }

    const float4 positionInvMass = g_RigidBodyPositionsInvMass[idx];
    const float4 orientation = QuaternionNormalize(g_RigidBodyOrientations[idx]);
    float4 linearVelocity = g_RigidBodyLinearVelocities[idx];
    float4 angularVelocity = g_RigidBodyAngularVelocities[idx];
    const uint bodyType = g_RigidBodyTypes[idx];
    const float4 kinematicTargetPosition = g_RigidBodyKinematicTargetPositions[idx];
    const float4 kinematicTargetOrientation =
        QuaternionNormalize(g_RigidBodyKinematicTargetOrientations[idx]);
    const bool kinematicTargetEnabled =
        (g_RigidBodyKinematicTargetFlags[idx] & kKinematicTargetEnabled) != 0u;

    g_PreviousRigidBodyPositionsInvMass[idx] = positionInvMass;
    g_PreviousRigidBodyOrientations[idx] = orientation;

    if (bodyType == 0u)
    {
        g_PredictedRigidBodyPositionsInvMass[idx] = positionInvMass;
        g_PredictedRigidBodyOrientations[idx] = orientation;
        g_PredictedRigidBodyLinearVelocities[idx] = float4(0.0, 0.0, 0.0, 0.0);
        g_PredictedRigidBodyAngularVelocities[idx] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    if (bodyType == 1u)
    {
        const float3 predictedPosition =
            kinematicTargetEnabled ? kinematicTargetPosition.xyz : positionInvMass.xyz;
        const float4 predictedOrientation =
            kinematicTargetEnabled ? kinematicTargetOrientation : orientation;
        linearVelocity.xyz = (predictedPosition - positionInvMass.xyz) / max(dt, kEpsilon);
        angularVelocity.xyz =
            AngularVelocityFromQuaternionDelta(orientation, predictedOrientation, max(dt, kEpsilon));

        g_PredictedRigidBodyPositionsInvMass[idx] =
            float4(predictedPosition, positionInvMass.w);
        g_PredictedRigidBodyOrientations[idx] = predictedOrientation;
        g_PredictedRigidBodyLinearVelocities[idx] = linearVelocity;
        g_PredictedRigidBodyAngularVelocities[idx] = angularVelocity;
        return;
    }

    linearVelocity.xyz += kGravity * dt;
    const float3 predictedPosition = positionInvMass.xyz + linearVelocity.xyz * dt;
    const float4 predictedOrientation =
        IntegrateOrientation(orientation, angularVelocity.xyz, dt);

    g_PredictedRigidBodyPositionsInvMass[idx] =
        float4(predictedPosition, positionInvMass.w);
    g_PredictedRigidBodyOrientations[idx] = predictedOrientation;
    g_PredictedRigidBodyLinearVelocities[idx] = linearVelocity;
    g_PredictedRigidBodyAngularVelocities[idx] = angularVelocity;
}
