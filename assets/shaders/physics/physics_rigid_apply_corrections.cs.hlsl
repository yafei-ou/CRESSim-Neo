cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint pairCount;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
    uint reserved0;
    uint reserved1;
};

#include "physics/physics_rigid_common.hlsli"

RWStructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_PredictedRigidBodyOrientations;
RWStructuredBuffer<float4> g_RigidBodyTranslationCorrections;
RWStructuredBuffer<float4> g_RigidBodyRotationCorrections;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    float4 positionInvMass = g_PredictedRigidBodyPositionsInvMass[bodyIndex];
    float4 orientation = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyIndex]);
    const float3 translationCorrection = g_RigidBodyTranslationCorrections[bodyIndex].xyz;
    const float3 rotationCorrection = g_RigidBodyRotationCorrections[bodyIndex].xyz;

    if (positionInvMass.w != 0.0)
    {
        positionInvMass.xyz += translationCorrection;
        orientation =
            QuaternionNormalize(QuaternionMul(QuaternionFromRotationVector(rotationCorrection),
                                              orientation));
    }

    g_PredictedRigidBodyPositionsInvMass[bodyIndex] = positionInvMass;
    g_PredictedRigidBodyOrientations[bodyIndex] = orientation;
    g_RigidBodyTranslationCorrections[bodyIndex] = 0.0;
    g_RigidBodyRotationCorrections[bodyIndex] = 0.0;
}
