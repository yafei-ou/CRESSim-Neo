#include "physics/physics_rigid_dispatch_constants.hlsli"
#include "physics/physics_atomic_float.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/core/physics_math.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

static const float kMaxTotalTranslationCorrectionPerIter = 0.01;
static const float kMaxTotalRotationCorrectionPerIter = 0.10;

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
    float3 translationCorrection =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_RigidBodyTranslationCorrections, bodyIndex);
    float3 rotationCorrection =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_RigidBodyRotationCorrections, bodyIndex);

    if (bodyType == kRigidBodyTypeDynamic && positionInvMass.w != 0.0)
    {
        const float translationLength = length(translationCorrection);
        if (translationLength > kMaxTotalTranslationCorrectionPerIter && translationLength > kEpsilon)
        {
            translationCorrection *= kMaxTotalTranslationCorrectionPerIter / translationLength;
        }

        const float rotationLength = length(rotationCorrection);
        if (rotationLength > kMaxTotalRotationCorrectionPerIter && rotationLength > kEpsilon)
        {
            rotationCorrection *= kMaxTotalRotationCorrectionPerIter / rotationLength;
        }

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
