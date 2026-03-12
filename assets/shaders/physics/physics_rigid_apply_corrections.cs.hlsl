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

static const float kCorrectionAtomicScale = 100000.0;

RWStructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
RWStructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<uint> g_RigidBodyTypes;
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
    const uint bodyType = g_RigidBodyTypes[bodyIndex];
    const float3 translationCorrection =
        float3(g_RigidBodyTranslationCorrections[bodyIndex].xyz) / kCorrectionAtomicScale;
    const float3 rotationCorrection =
        float3(g_RigidBodyRotationCorrections[bodyIndex].xyz) / kCorrectionAtomicScale;

    if (bodyType == 2u && positionInvMass.w != 0.0)
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
