#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

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
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_RigidBodyTranslationCorrections, bodyIndex);
    const float3 rotationCorrection =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_RigidBodyRotationCorrections, bodyIndex);

    if (bodyType == kRigidBodyTypeDynamic && positionInvMass.w != 0.0)
    {
        positionInvMass.xyz += translationCorrection;
        orientation =
            QuaternionNormalize(QuaternionMul(QuaternionFromRotationVector(rotationCorrection),
                                              orientation));
    }

    CRESSIM_SB_STORE(g_PredictedRigidBodyPositionsInvMass, bodyIndex, positionInvMass);
    CRESSIM_SB_STORE(g_PredictedRigidBodyOrientations, bodyIndex, orientation);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyTranslationCorrections, bodyIndex);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_RigidBodyRotationCorrections, bodyIndex);
}
