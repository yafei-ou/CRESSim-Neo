#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

static const float kCorrectionAtomicScale = 100000.0;

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyTranslationCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    float4 positionInvMass = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyIndex);
    float4 orientation = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyIndex));
    const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyIndex);
    const float3 translationCorrection =
        float3(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, bodyIndex).xyz) / kCorrectionAtomicScale;
    const float3 rotationCorrection =
        float3(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, bodyIndex).xyz) / kCorrectionAtomicScale;

    if (bodyType == kRigidBodyTypeDynamic && positionInvMass.w != 0.0)
    {
        positionInvMass.xyz += translationCorrection;
        orientation =
            QuaternionNormalize(QuaternionMul(QuaternionFromRotationVector(rotationCorrection),
                                              orientation));
    }

    CRESSIM_SB_STORE(g_PredictedRigidBodyPositionsInvMass, bodyIndex, positionInvMass);
    CRESSIM_SB_STORE(g_PredictedRigidBodyOrientations, bodyIndex, orientation);
    CRESSIM_SB_STORE(g_RigidBodyTranslationCorrections, bodyIndex, int4(0, 0, 0, 0));
    CRESSIM_SB_STORE(g_RigidBodyRotationCorrections, bodyIndex, int4(0, 0, 0, 0));
}
