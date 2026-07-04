#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

// We still relax applied corrections for stability, but cache the same relaxed lambda that is
// actually applied so the XPBD compliance term stays internally consistent.
static const float kJointRelaxation = 0.7;
static const float kMaxJointError = 0.05;
static const float kMaxJointTranslationCorrection = 0.02;
static const float kMaxJointAngularCorrection = 0.12;
static const float kSliderTranslationRegularization = 1e-5;
static const float kSliderAngularRegularization = 5e-5;
static const float kMinXpbdDt = 1e-5;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);
CRESSIM_STRUCTURED_BUFFER(uint, g_SliderJointIndices);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SliderJointLambdas0123);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SliderJointLambdas45);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint jointListIndex = dispatchThreadID.x;
    if (jointListIndex >= jointCount)
    {
        return;
    }

    const uint jointIndex = CRESSIM_SB_LOAD(g_SliderJointIndices, jointListIndex);
    const GpuSliderJoint joint = CRESSIM_SB_LOAD(g_SliderJoints, jointIndex);
    if (joint.enabled == 0u)
    {
        return;
    }

    const uint bodyA = joint.bodyA;
    const uint bodyB = joint.bodyB;
    if (bodyA >= rigidBodyCount || bodyB >= rigidBodyCount)
    {
        return;
    }

    const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyA);
    const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyB);
    const float4 posInvMassA = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyA);
    const float4 posInvMassB = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyB);
    const float invMassA = bodyTypeA == kRigidBodyTypeDynamic ? posInvMassA.w : 0.0;
    const float invMassB = bodyTypeB == kRigidBodyTypeDynamic ? posInvMassB.w : 0.0;
    if (invMassA == 0.0 && invMassB == 0.0)
    {
        return;
    }

    const float4 qA = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyA));
    const float4 qB = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyB));
    float3 invInertiaA = CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, bodyA).xyz;
    float3 invInertiaB = CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, bodyB).xyz;
    if (invMassA == 0.0)
        invInertiaA = 0.0;
    if (invMassB == 0.0)
        invInertiaB = 0.0;

    const float3 rA = QuaternionRotate(qA, joint.localAnchorA.xyz);
    const float3 rB = QuaternionRotate(qB, joint.localAnchorB.xyz);
    const float3 a0 = SafeNormalize(QuaternionRotate(qA, joint.localAxisA0.xyz), float3(1.0, 0.0, 0.0));
    const float3 a1 = SafeNormalize(QuaternionRotate(qA, joint.localAxisA1.xyz), ChoosePerpendicular(a0));
    const float3 a2 = SafeNormalize(QuaternionRotate(qA, joint.localAxisA2.xyz), normalize(cross(a0, a1)));
    const float3 pA = posInvMassA.xyz + rA;
    const float3 pB = posInvMassB.xyz + rB;
    const float3 rawDelta = pA - pB;
    const float3 delta = ClampErrorVector(rawDelta, kMaxJointError);
    const float3 tRow0 = ComputeProjectionJacobianRow(joint.projectionRow0, qA, qB);
    const float3 tRow1 = ComputeProjectionJacobianRow(joint.projectionRow1, qA, qB);
    const float3 tRow2 = ComputeProjectionJacobianRow(joint.projectionRow2, qA, qB);
    const bool limitEnabled = joint.limitParams.x > 0.5;
    const float2 limitRange = joint.limitParams.yz;
    const float invDtSq = 1.0 / max(dt * dt, kMinXpbdDt * kMinXpbdDt);
    const float constraintAlpha = max(joint.limitParams.w, 0.0) * invDtSq;
    const float constraintScale = kJointRelaxation;
    const float limitScale = kJointRelaxation;
    const float currentPosition = joint.driveTargetParams.y - dot(rawDelta, a0);
    float limitTargetPosition = currentPosition;
    const bool limitActive =
        limitEnabled && ComputeLimitTarget(currentPosition, limitRange, limitTargetPosition);
    const float4 lambdaState0123 = CRESSIM_SB_LOAD(g_SliderJointLambdas0123, jointIndex);
    const float4 lambdaState45 = CRESSIM_SB_LOAD(g_SliderJointLambdas45, jointIndex);
    float3 linearImpulse = 0.0;
    float3 angularImpulseA = 0.0;
    float3 angularImpulseB = 0.0;

    float3 jLinA[5];
    float3 jAngA[5];
    float3 jLinB[5];
    float3 jAngB[5];
    jLinA[0] = a1;
    jLinA[1] = a2;
    jLinA[2] = 0.0;
    jLinA[3] = 0.0;
    jLinA[4] = 0.0;
    jLinB[0] = -a1;
    jLinB[1] = -a2;
    jLinB[2] = 0.0;
    jLinB[3] = 0.0;
    jLinB[4] = 0.0;
    jAngA[0] = cross(rA, a1);
    jAngA[1] = cross(rA, a2);
    jAngA[2] = tRow0;
    jAngA[3] = tRow1;
    jAngA[4] = tRow2;
    jAngB[0] = -cross(rB, a1);
    jAngB[1] = -cross(rB, a2);
    jAngB[2] = -tRow0;
    jAngB[3] = -tRow1;
    jAngB[4] = -tRow2;

    float previousLambda[5];
    previousLambda[0] = lambdaState0123.x;
    previousLambda[1] = lambdaState0123.y;
    previousLambda[2] = lambdaState0123.z;
    previousLambda[3] = lambdaState0123.w;
    previousLambda[4] = lambdaState45.x;

    float rhs[5];
    rhs[0] = -dot(delta, a1) - constraintAlpha * previousLambda[0];
    rhs[1] = -dot(delta, a2) - constraintAlpha * previousLambda[1];
    rhs[2] = -ComputeProjectionConstraintValue(joint.projectionRow0, qA, qB) -
             constraintAlpha * previousLambda[2];
    rhs[3] = -ComputeProjectionConstraintValue(joint.projectionRow1, qA, qB) -
             constraintAlpha * previousLambda[3];
    rhs[4] = -ComputeProjectionConstraintValue(joint.projectionRow2, qA, qB) -
             constraintAlpha * previousLambda[4];

    float k[5][5];
    [unroll] for (uint row = 0u; row < 5u; ++row)
    {
        [unroll] for (uint col = 0u; col < 5u; ++col)
        {
            k[row][col] = ComputeConstraintMatrixElement(
                invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                jLinA[row], jAngA[row], jLinB[row], jAngB[row],
                jLinA[col], jAngA[col], jLinB[col], jAngB[col]);
        }
    }

    // Regularize the slider system to reduce sensitivity in near-symmetric
    // frame configurations while preserving the fully coupled 5x5 structure.
    k[0][0] += kSliderTranslationRegularization + constraintAlpha;
    k[1][1] += kSliderTranslationRegularization + constraintAlpha;
    k[2][2] += kSliderAngularRegularization + constraintAlpha;
    k[3][3] += kSliderAngularRegularization + constraintAlpha;
    k[4][4] += kSliderAngularRegularization + constraintAlpha;

    float deltaLambda[5];
    if (!SolveLinearSystem5x5(k, rhs, deltaLambda))
    {
        return;
    }

    float appliedDeltaLambda[5];
    [unroll] for (uint row = 0u; row < 5u; ++row)
    {
        appliedDeltaLambda[row] = deltaLambda[row] * constraintScale;
        linearImpulse += jLinA[row] * appliedDeltaLambda[row];
        angularImpulseA += jAngA[row] * appliedDeltaLambda[row];
        angularImpulseB += jAngB[row] * appliedDeltaLambda[row];
    }

    CRESSIM_SB_STORE(g_SliderJointLambdas0123, jointIndex,
                     float4(previousLambda[0] + appliedDeltaLambda[0],
                            previousLambda[1] + appliedDeltaLambda[1],
                            previousLambda[2] + appliedDeltaLambda[2],
                            previousLambda[3] + appliedDeltaLambda[3]));
    CRESSIM_SB_STORE(g_SliderJointLambdas45, jointIndex,
                     float4(previousLambda[4] + appliedDeltaLambda[4], 0.0, lambdaState45.z,
                            lambdaState45.w));

    if (limitActive)
    {
        const float previousLimitLambda = lambdaState45.y;
        const float limitError = ClampErrorScalar(
            dot(rawDelta, a0) - joint.driveTargetParams.y + limitTargetPosition,
            kMaxJointTranslationCorrection);
        const float3 jLinLimitA = a0;
        const float3 jLinLimitB = -a0;
        const float3 jAngLimitA = cross(rA, a0);
        const float3 jAngLimitB = -cross(rB, a0);
        const float rhs = -limitError - constraintAlpha * previousLimitLambda;
        const float limitK = ComputeConstraintMatrixElement(
                                 invMassA, invInertiaA, qA, invMassB, invInertiaB, qB, jLinLimitA,
                                 jAngLimitA, jLinLimitB, jAngLimitB, jLinLimitA, jAngLimitA,
                                 jLinLimitB, jAngLimitB) +
                             kSliderTranslationRegularization + constraintAlpha;
        if (abs(limitK) > kEpsilon)
        {
            const float appliedDeltaLambda = (rhs / limitK) * limitScale;
            linearImpulse += jLinLimitA * appliedDeltaLambda;
            angularImpulseA += jAngLimitA * appliedDeltaLambda;
            angularImpulseB += jAngLimitB * appliedDeltaLambda;
            CRESSIM_SB_STORE(g_SliderJointLambdas45, jointIndex,
                             float4(lambdaState45.x, previousLimitLambda + appliedDeltaLambda,
                                    lambdaState45.z, lambdaState45.w));
        }
    }

    const float3 translationA = linearImpulse * invMassA;
    const float3 rotationA = MultiplyWorldInverseInertia(invInertiaA, qA, angularImpulseA);
    const float3 translationB = -linearImpulse * invMassB;
    const float3 rotationB = MultiplyWorldInverseInertia(invInertiaB, qB, angularImpulseB);
    const float correctionScale =
        ComputeCorrectionLimitScale(translationA, rotationA, translationB, rotationB,
                                    kMaxJointTranslationCorrection, kMaxJointAngularCorrection);
    const float3 limitedTranslationA = translationA * correctionScale;
    const float3 limitedRotationA = rotationA * correctionScale;
    const float3 limitedTranslationB = translationB * correctionScale;
    const float3 limitedRotationB = rotationB * correctionScale;

    if (bodyTypeA == kRigidBodyTypeDynamic && invMassA != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyA,
                                      limitedTranslationA);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyA, limitedRotationA);
    }
    if (bodyTypeB == kRigidBodyTypeDynamic && invMassB != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyB,
                                      limitedTranslationB);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyB, limitedRotationB);
    }
}
