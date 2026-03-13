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

#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_PreviousRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PreviousRigidBodyOrientations;
StructuredBuffer<uint> g_RigidBodyTypes;

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
    const uint bodyType = g_RigidBodyTypes[bodyIndex];
    float4 predictedPositionInvMass = g_PredictedRigidBodyPositionsInvMass[bodyIndex];
    float4 predictedOrientation =
        QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyIndex]);

    if (bodyType == 0u)
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
