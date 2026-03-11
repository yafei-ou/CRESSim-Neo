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

static const float kCorrectionAtomicScale = 100000.0;

RWStructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_PredictedRigidBodyOrientations;
RWStructuredBuffer<int4> g_RigidBodyTranslationCorrections;
RWStructuredBuffer<int4> g_RigidBodyRotationCorrections;

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
    const float3 translationCorrection =
        float3(g_RigidBodyTranslationCorrections[bodyIndex].xyz) / kCorrectionAtomicScale;
    const float3 rotationCorrection =
        float3(g_RigidBodyRotationCorrections[bodyIndex].xyz) / kCorrectionAtomicScale;

    if (positionInvMass.w != 0.0)
    {
        positionInvMass.xyz += translationCorrection;
        orientation =
            QuaternionNormalize(QuaternionMul(QuaternionFromRotationVector(rotationCorrection),
                                              orientation));
    }

    g_PredictedRigidBodyPositionsInvMass[bodyIndex] = positionInvMass;
    g_PredictedRigidBodyOrientations[bodyIndex] = orientation;
    g_RigidBodyTranslationCorrections[bodyIndex] = int4(0, 0, 0, 0);
    g_RigidBodyRotationCorrections[bodyIndex] = int4(0, 0, 0, 0);
}
