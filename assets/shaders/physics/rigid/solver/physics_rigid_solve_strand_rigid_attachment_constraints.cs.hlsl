#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/structured_buffer_compat.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

static const float kAttachmentRelaxation = 0.95;
static const float kMaxAttachmentTranslationCorrection = 0.05;
static const float kMaxAttachmentAngularCorrection = 0.2;

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(GpuStrandRigidAttachmentConstraint, g_StrandRigidAttachments);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandRigidAttachmentLambda, g_StrandRigidAttachmentLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandRigidAttachmentCorrection, g_StrandRigidAttachmentCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);

float ComputeStrandOrientationInvMass(float wA, float wB, float restLength)
{
    const float dynamicWeightSum = max(wA + wB, 0.0);
    if (dynamicWeightSum <= kEpsilon || restLength <= kEpsilon)
    {
        return 0.0;
    }

    const float averageInvMass = dynamicWeightSum * 0.5;
    return 12.0 * averageInvMass / max(restLength * restLength, kEpsilon);
}

float ComputeLimitScale(float3 value, float maxMagnitude)
{
    if (maxMagnitude <= 0.0)
    {
        return 1.0;
    }

    const float lengthSq = dot(value, value);
    if (lengthSq <= maxMagnitude * maxMagnitude || lengthSq <= kEpsilon)
    {
        return 1.0;
    }

    return maxMagnitude * rsqrt(lengthSq);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint constraintIndex = dispatchThreadID.x;
    if (constraintIndex >= reserved0)
    {
        return;
    }

    const GpuStrandRigidAttachmentConstraint constraint =
        CRESSIM_SB_LOAD(g_StrandRigidAttachments, constraintIndex);
    if (constraint.rigidBodyIndex >= rigidBodyCount)
    {
        return;
    }

    const GpuStrandSegment segment = CRESSIM_SB_LOAD(g_StrandSegments, constraint.segmentIndex);
    if (segment.particleA >= reserved1 || segment.particleB >= reserved1)
    {
        return;
    }

    const float4 particleA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleA);
    const float4 particleB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleB);
    const float4 rigidPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, constraint.rigidBodyIndex);

    const float t = saturate(constraint.segmentT);
    const float weightA = 1.0 - t;
    const float weightB = t;
    const float invMassA = particleA.w;
    const float invMassB = particleB.w;
    const float strandStationInvMass = invMassA * weightA * weightA + invMassB * weightB * weightB;
    if (strandStationInvMass <= kEpsilon)
    {
        return;
    }

    const float4 rigidOrientation = QuaternionNormalize(
        CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, constraint.rigidBodyIndex));
    const float3 worldLeverArm = QuaternionRotate(rigidOrientation, constraint.localAnchor.xyz);
    const float3 rigidAnchorPosition = rigidPositionInvMass.xyz + worldLeverArm;
    const float3 strandStationPosition = lerp(particleA.xyz, particleB.xyz, t);

    GpuStrandRigidAttachmentLambda lambdaState =
        CRESSIM_SB_LOAD(g_StrandRigidAttachmentLambdas, constraintIndex);
    float3 translationLambda = lambdaState.translation.xyz;
    float3 rotationLambda = lambdaState.rotation.xyz;

    float3 particleCorrectionA = 0.0;
    float3 particleCorrectionB = 0.0;
    float3 segmentRotationCorrection = 0.0;

    const float3 translationConstraint = strandStationPosition - rigidAnchorPosition;
    const float translationAlpha =
        max(constraint.translationCompliance, 0.0) / max(dt * dt, kEpsilon);

    [unroll] for (uint axisIndex = 0u; axisIndex < 3u; ++axisIndex)
    {
        const float3 axis = axisIndex == 0u ? float3(1.0, 0.0, 0.0)
                          : axisIndex == 1u ? float3(0.0, 1.0, 0.0)
                                             : float3(0.0, 0.0, 1.0);
        const float c = dot(translationConstraint, axis);
        const float denominator = strandStationInvMass + translationAlpha;
        if (denominator <= kEpsilon)
        {
            continue;
        }

        const float deltaLambda = -(c + translationAlpha * translationLambda[axisIndex]) / denominator;
        translationLambda[axisIndex] += deltaLambda;
        particleCorrectionA += axis * (deltaLambda * invMassA * weightA * kAttachmentRelaxation);
        particleCorrectionB += axis * (deltaLambda * invMassB * weightB * kAttachmentRelaxation);
    }

    const float4 segmentOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_StrandSegmentStates, constraint.segmentIndex).orientation);
    const float4 targetOrientation =
        QuaternionNormalize(QuaternionMul(rigidOrientation, QuaternionNormalize(constraint.localRotation)));
    float4 rotationError =
        QuaternionNormalize(QuaternionMul(targetOrientation, QuaternionConjugate(segmentOrientation)));
    if (rotationError.w < 0.0)
    {
        rotationError = -rotationError;
    }

    const float3 rotationConstraint = 2.0 * rotationError.xyz;
    const float rotationAlpha = max(constraint.rotationCompliance, 0.0) / max(dt * dt, kEpsilon);
    const float orientationInvMass =
        ComputeStrandOrientationInvMass(invMassA, invMassB, segment.restLength);

    [unroll] for (uint axisIndex = 0u; axisIndex < 3u; ++axisIndex)
    {
        const float3 axis = axisIndex == 0u ? float3(1.0, 0.0, 0.0)
                          : axisIndex == 1u ? float3(0.0, 1.0, 0.0)
                                             : float3(0.0, 0.0, 1.0);
        const float c = dot(rotationConstraint, axis);
        const float denominator = orientationInvMass + rotationAlpha;
        if (denominator <= kEpsilon)
        {
            continue;
        }

        const float deltaLambda = -(c + rotationAlpha * rotationLambda[axisIndex]) / denominator;
        rotationLambda[axisIndex] += deltaLambda;
        segmentRotationCorrection +=
            axis * (deltaLambda * orientationInvMass * kAttachmentRelaxation);
    }

    lambdaState.translation = float4(translationLambda, 0.0);
    lambdaState.rotation    = float4(rotationLambda, 0.0);
    CRESSIM_SB_STORE(g_StrandRigidAttachmentLambdas, constraintIndex, lambdaState);

    float scale = 1.0;
    scale = min(scale, ComputeLimitScale(particleCorrectionA, kMaxAttachmentTranslationCorrection));
    scale = min(scale, ComputeLimitScale(particleCorrectionB, kMaxAttachmentTranslationCorrection));
    scale = min(scale, ComputeLimitScale(segmentRotationCorrection, kMaxAttachmentAngularCorrection));

    if (invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, segment.particleA,
                                      particleCorrectionA * scale);
    }
    if (invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, segment.particleB,
                                      particleCorrectionB * scale);
    }
    GpuStrandRigidAttachmentCorrection correction;
    correction.segmentRotation = float4(segmentRotationCorrection * scale, 0.0);
    CRESSIM_SB_STORE(g_StrandRigidAttachmentCorrections, constraintIndex, correction);
}
