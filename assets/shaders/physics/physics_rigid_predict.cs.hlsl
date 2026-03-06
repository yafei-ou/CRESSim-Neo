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

StructuredBuffer<float4> g_RigidBodyPositionsInvMass;
StructuredBuffer<float4> g_RigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyLinearVelocities;
StructuredBuffer<float4> g_RigidBodyAngularVelocities;

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
    const float4 angularVelocity = g_RigidBodyAngularVelocities[idx];

    g_PreviousRigidBodyPositionsInvMass[idx] = positionInvMass;
    g_PreviousRigidBodyOrientations[idx] = orientation;

    if (positionInvMass.w == 0.0)
    {
        g_PredictedRigidBodyPositionsInvMass[idx] = positionInvMass;
        g_PredictedRigidBodyOrientations[idx] = orientation;
        g_PredictedRigidBodyLinearVelocities[idx] = float4(0.0, 0.0, 0.0, 0.0);
        g_PredictedRigidBodyAngularVelocities[idx] = float4(0.0, 0.0, 0.0, 0.0);
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
