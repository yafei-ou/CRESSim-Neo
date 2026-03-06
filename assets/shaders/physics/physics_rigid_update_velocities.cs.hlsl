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

StructuredBuffer<float4> g_PreviousRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PreviousRigidBodyOrientations;

RWStructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_PredictedRigidBodyOrientations;
RWStructuredBuffer<float4> g_PredictedRigidBodyLinearVelocities;
RWStructuredBuffer<float4> g_PredictedRigidBodyAngularVelocities;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    const float4 previousPositionInvMass = g_PreviousRigidBodyPositionsInvMass[bodyIndex];
    const float4 previousOrientation =
        QuaternionNormalize(g_PreviousRigidBodyOrientations[bodyIndex]);
    float4 predictedPositionInvMass = g_PredictedRigidBodyPositionsInvMass[bodyIndex];
    float4 predictedOrientation =
        QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyIndex]);

    if (predictedPositionInvMass.w == 0.0)
    {
        predictedOrientation = QuaternionNormalize(predictedOrientation);
        g_PredictedRigidBodyPositionsInvMass[bodyIndex] = predictedPositionInvMass;
        g_PredictedRigidBodyOrientations[bodyIndex] = predictedOrientation;
        g_PredictedRigidBodyLinearVelocities[bodyIndex] = 0.0;
        g_PredictedRigidBodyAngularVelocities[bodyIndex] = 0.0;
        return;
    }

    const float3 linearVelocity =
        (predictedPositionInvMass.xyz - previousPositionInvMass.xyz) / max(dt, kEpsilon);
    const float3 angularVelocity =
        AngularVelocityFromQuaternionDelta(previousOrientation, predictedOrientation,
                                           max(dt, kEpsilon));

    g_PredictedRigidBodyPositionsInvMass[bodyIndex] = predictedPositionInvMass;
    g_PredictedRigidBodyOrientations[bodyIndex] = predictedOrientation;
    g_PredictedRigidBodyLinearVelocities[bodyIndex] = float4(linearVelocity, 0.0);
    g_PredictedRigidBodyAngularVelocities[bodyIndex] = float4(angularVelocity, 0.0);
}
