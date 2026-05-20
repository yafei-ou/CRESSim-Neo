#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

// We still relax applied corrections for stability, but cache the same relaxed lambda that is
// actually applied so the XPBD compliance term stays internally consistent.
static const float kJointRelaxation = 0.7;
static const float kMaxJointError = 0.05;
static const float kMaxJointDriveAngularError = 1.0;
static const float kMaxJointTranslationCorrection = 0.02;
static const float kMaxJointAngularCorrection = 0.12;
static const float kJointDriveRelaxation = 0.2;
static const float kHingeTranslationRegularization = 1e-5;
static const float kHingeAngularRegularization = 5e-5;
static const float kMinXpbdDt = 1e-5;

#ifndef CRESSIM_JOINT_DRIVE_MODE_TARGET_POSITION
#define CRESSIM_JOINT_DRIVE_MODE_TARGET_POSITION 0
#endif

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuHingeJoint, g_HingeJoints);
CRESSIM_STRUCTURED_BUFFER(uint, g_HingeJointIndices);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_HingeJointLambdas0123);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_HingeJointLambdas45);

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

    const uint jointIndex = CRESSIM_SB_LOAD(g_HingeJointIndices, jointListIndex);
    const GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, jointIndex);
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
    const float3 pA = posInvMassA.xyz + rA;
    const float3 pB = posInvMassB.xyz + rB;
    const float3 delta = ClampErrorVector(pA - pB, kMaxJointError);

    const float3 tRow0 = ComputeProjectionJacobianRow(joint.projectionRow0, qA, qB);
    const float3 tRow1 = ComputeProjectionJacobianRow(joint.projectionRow1, qA, qB);
    const float3 tRow2 = ComputeProjectionJacobianRow(joint.projectionRow2, qA, qB);
    const bool limitEnabled = joint.limitParams.x > 0.5;
    const float2 limitRange = joint.limitParams.yz;
    const float currentAngle = ComputeHingeAngle(joint.projectionRow0, qA, qB);

    float limitTargetAngle = currentAngle;
    const bool limitActive = limitEnabled && ComputeLimitTarget(currentAngle, limitRange, limitTargetAngle);
    const float driveTargetAngle =
        limitEnabled ? clamp(joint.driveTargetParams.x, limitRange.x, limitRange.y)
                     : joint.driveTargetParams.x;
    const float driveError = currentAngle - driveTargetAngle;

    const float invDtSq = 1.0 / max(dt * dt, kMinXpbdDt * kMinXpbdDt);
    const float constraintAlpha = max(joint.limitParams.w, 0.0) * invDtSq;
    const float driveAlpha = max(joint.driveTargetParams.z, 0.0) * invDtSq;
    const float constraintScale = kJointRelaxation;
    const float driveScale = kJointDriveRelaxation;
    const float limitScale = kJointRelaxation;
    const float4 lambdaState0123 = CRESSIM_SB_LOAD(g_HingeJointLambdas0123, jointIndex);
    const float4 lambdaState45 = CRESSIM_SB_LOAD(g_HingeJointLambdas45, jointIndex);

#if CRESSIM_JOINT_DRIVE_MODE_TARGET_POSITION
    float3 jLinA[6];
    float3 jAngA[6];
    float3 jLinB[6];
    float3 jAngB[6];
    jLinA[0] = float3(1.0, 0.0, 0.0);
    jLinA[1] = float3(0.0, 1.0, 0.0);
    jLinA[2] = float3(0.0, 0.0, 1.0);
    jLinA[3] = 0.0;
    jLinA[4] = 0.0;
    jLinA[5] = 0.0;
    jLinB[0] = -jLinA[0];
    jLinB[1] = -jLinA[1];
    jLinB[2] = -jLinA[2];
    jLinB[3] = 0.0;
    jLinB[4] = 0.0;
    jLinB[5] = 0.0;
    jAngA[0] = cross(rA, jLinA[0]);
    jAngA[1] = cross(rA, jLinA[1]);
    jAngA[2] = cross(rA, jLinA[2]);
    jAngA[3] = tRow0;
    jAngA[4] = tRow1;
    jAngA[5] = tRow2;
    jAngB[0] = -cross(rB, jLinA[0]);
    jAngB[1] = -cross(rB, jLinA[1]);
    jAngB[2] = -cross(rB, jLinA[2]);
    jAngB[3] = -tRow0;
    jAngB[4] = -tRow1;
    jAngB[5] = -tRow2;

    float previousLambda[6];
    previousLambda[0] = lambdaState0123.x;
    previousLambda[1] = lambdaState0123.y;
    previousLambda[2] = lambdaState0123.z;
    previousLambda[3] = lambdaState0123.w;
    previousLambda[4] = lambdaState45.x;
    previousLambda[5] = lambdaState45.y;

    float rhs[6];
    rhs[0] = -delta.x - constraintAlpha * previousLambda[0];
    rhs[1] = -delta.y - constraintAlpha * previousLambda[1];
    rhs[2] = -delta.z - constraintAlpha * previousLambda[2];
    rhs[3] = -sin(0.5 * ClampErrorScalar(driveError, kMaxJointDriveAngularError)) -
             driveAlpha * previousLambda[3];
    rhs[4] = -ComputeProjectionConstraintValue(joint.projectionRow1, qA, qB) -
             constraintAlpha * previousLambda[4];
    rhs[5] = -ComputeProjectionConstraintValue(joint.projectionRow2, qA, qB) -
             constraintAlpha * previousLambda[5];

    float k[6][6];
    [unroll] for (uint row = 0u; row < 6u; ++row)
    {
        [unroll] for (uint col = 0u; col < 6u; ++col)
        {
            k[row][col] = ComputeConstraintMatrixElement(
                invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                jLinA[row], jAngA[row], jLinB[row], jAngB[row],
                jLinA[col], jAngA[col], jLinB[col], jAngB[col]);
        }
    }

    k[0][0] += kHingeTranslationRegularization + constraintAlpha;
    k[1][1] += kHingeTranslationRegularization + constraintAlpha;
    k[2][2] += kHingeTranslationRegularization + constraintAlpha;
    k[3][3] += kHingeAngularRegularization + driveAlpha;
    k[4][4] += kHingeAngularRegularization + constraintAlpha;
    k[5][5] += kHingeAngularRegularization + constraintAlpha;

    float deltaLambda[6];
    if (!SolveLinearSystem6x6(k, rhs, deltaLambda))
    {
        return;
    }

    float3 linearImpulse = 0.0;
    float3 angularImpulseA = 0.0;
    float3 angularImpulseB = 0.0;
    float appliedDeltaLambda[6];
    [unroll] for (uint row = 0u; row < 6u; ++row)
    {
        appliedDeltaLambda[row] = deltaLambda[row] * (row == 3u ? driveScale : constraintScale);
        linearImpulse += jLinA[row] * appliedDeltaLambda[row];
        angularImpulseA += jAngA[row] * appliedDeltaLambda[row];
        angularImpulseB += jAngB[row] * appliedDeltaLambda[row];
    }

    CRESSIM_SB_STORE(g_HingeJointLambdas0123, jointIndex,
                     float4(previousLambda[0] + appliedDeltaLambda[0],
                            previousLambda[1] + appliedDeltaLambda[1],
                            previousLambda[2] + appliedDeltaLambda[2],
                            previousLambda[3] + appliedDeltaLambda[3]));
    CRESSIM_SB_STORE(g_HingeJointLambdas45, jointIndex,
                     float4(previousLambda[4] + appliedDeltaLambda[4],
                            previousLambda[5] + appliedDeltaLambda[5], lambdaState45.z,
                            lambdaState45.w));
#else
    float3 linearImpulse = 0.0;
    float3 angularImpulseA = 0.0;
    float3 angularImpulseB = 0.0;

    if (limitActive)
    {
        float3 jLinA[6];
        float3 jAngA[6];
        float3 jLinB[6];
        float3 jAngB[6];
        jLinA[0] = float3(1.0, 0.0, 0.0);
        jLinA[1] = float3(0.0, 1.0, 0.0);
        jLinA[2] = float3(0.0, 0.0, 1.0);
        jLinA[3] = 0.0;
        jLinA[4] = 0.0;
        jLinA[5] = 0.0;
        jLinB[0] = -jLinA[0];
        jLinB[1] = -jLinA[1];
        jLinB[2] = -jLinA[2];
        jLinB[3] = 0.0;
        jLinB[4] = 0.0;
        jLinB[5] = 0.0;
        jAngA[0] = cross(rA, jLinA[0]);
        jAngA[1] = cross(rA, jLinA[1]);
        jAngA[2] = cross(rA, jLinA[2]);
        jAngA[3] = tRow0;
        jAngA[4] = tRow1;
        jAngA[5] = tRow2;
        jAngB[0] = -cross(rB, jLinA[0]);
        jAngB[1] = -cross(rB, jLinA[1]);
        jAngB[2] = -cross(rB, jLinA[2]);
        jAngB[3] = -tRow0;
        jAngB[4] = -tRow1;
        jAngB[5] = -tRow2;

        float previousLambda[6];
        previousLambda[0] = lambdaState0123.x;
        previousLambda[1] = lambdaState0123.y;
        previousLambda[2] = lambdaState0123.z;
        previousLambda[3] = lambdaState0123.w;
        previousLambda[4] = lambdaState45.x;
        previousLambda[5] = lambdaState45.y;

        float rhs[6];
        rhs[0] = -delta.x - constraintAlpha * previousLambda[0];
        rhs[1] = -delta.y - constraintAlpha * previousLambda[1];
        rhs[2] = -delta.z - constraintAlpha * previousLambda[2];
        rhs[3] = -sin(0.5 * ClampErrorScalar(currentAngle - limitTargetAngle,
                                             kMaxJointAngularCorrection)) -
                 constraintAlpha * previousLambda[3];
        rhs[4] = -ComputeProjectionConstraintValue(joint.projectionRow1, qA, qB) -
                 constraintAlpha * previousLambda[4];
        rhs[5] = -ComputeProjectionConstraintValue(joint.projectionRow2, qA, qB) -
                 constraintAlpha * previousLambda[5];

        float k[6][6];
        [unroll] for (uint row = 0u; row < 6u; ++row)
        {
            [unroll] for (uint col = 0u; col < 6u; ++col)
            {
                k[row][col] = ComputeConstraintMatrixElement(
                    invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                    jLinA[row], jAngA[row], jLinB[row], jAngB[row],
                    jLinA[col], jAngA[col], jLinB[col], jAngB[col]);
            }
        }

        k[0][0] += kHingeTranslationRegularization + constraintAlpha;
        k[1][1] += kHingeTranslationRegularization + constraintAlpha;
        k[2][2] += kHingeTranslationRegularization + constraintAlpha;
        k[3][3] += kHingeAngularRegularization + constraintAlpha;
        k[4][4] += kHingeAngularRegularization + constraintAlpha;
        k[5][5] += kHingeAngularRegularization + constraintAlpha;

        float deltaLambda[6];
        if (!SolveLinearSystem6x6(k, rhs, deltaLambda))
        {
            return;
        }

        float appliedDeltaLambda[6];
        [unroll] for (uint row = 0u; row < 6u; ++row)
        {
            appliedDeltaLambda[row] = deltaLambda[row] * limitScale;
            linearImpulse += jLinA[row] * appliedDeltaLambda[row];
            angularImpulseA += jAngA[row] * appliedDeltaLambda[row];
            angularImpulseB += jAngB[row] * appliedDeltaLambda[row];
        }

        CRESSIM_SB_STORE(g_HingeJointLambdas0123, jointIndex,
                         float4(previousLambda[0] + appliedDeltaLambda[0],
                                previousLambda[1] + appliedDeltaLambda[1],
                                previousLambda[2] + appliedDeltaLambda[2],
                                previousLambda[3] + appliedDeltaLambda[3]));
        CRESSIM_SB_STORE(g_HingeJointLambdas45, jointIndex,
                         float4(previousLambda[4] + appliedDeltaLambda[4],
                                previousLambda[5] + appliedDeltaLambda[5], lambdaState45.z,
                                lambdaState45.w));
    }
    else
    {
        float3 jLinA[5];
        float3 jAngA[5];
        float3 jLinB[5];
        float3 jAngB[5];
        jLinA[0] = float3(1.0, 0.0, 0.0);
        jLinA[1] = float3(0.0, 1.0, 0.0);
        jLinA[2] = float3(0.0, 0.0, 1.0);
        jLinA[3] = 0.0;
        jLinA[4] = 0.0;
        jLinB[0] = -jLinA[0];
        jLinB[1] = -jLinA[1];
        jLinB[2] = -jLinA[2];
        jLinB[3] = 0.0;
        jLinB[4] = 0.0;
        jAngA[0] = cross(rA, jLinA[0]);
        jAngA[1] = cross(rA, jLinA[1]);
        jAngA[2] = cross(rA, jLinA[2]);
        jAngA[3] = tRow1;
        jAngA[4] = tRow2;
        jAngB[0] = -cross(rB, jLinA[0]);
        jAngB[1] = -cross(rB, jLinA[1]);
        jAngB[2] = -cross(rB, jLinA[2]);
        jAngB[3] = -tRow1;
        jAngB[4] = -tRow2;

        float previousLambda[5];
        previousLambda[0] = lambdaState0123.x;
        previousLambda[1] = lambdaState0123.y;
        previousLambda[2] = lambdaState0123.z;
        previousLambda[3] = lambdaState45.x;
        previousLambda[4] = lambdaState45.y;

        float rhs[5];
        rhs[0] = -delta.x - constraintAlpha * previousLambda[0];
        rhs[1] = -delta.y - constraintAlpha * previousLambda[1];
        rhs[2] = -delta.z - constraintAlpha * previousLambda[2];
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

        k[0][0] += kHingeTranslationRegularization + constraintAlpha;
        k[1][1] += kHingeTranslationRegularization + constraintAlpha;
        k[2][2] += kHingeTranslationRegularization + constraintAlpha;
        k[3][3] += kHingeAngularRegularization + constraintAlpha;
        k[4][4] += kHingeAngularRegularization + constraintAlpha;

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

        CRESSIM_SB_STORE(g_HingeJointLambdas0123, jointIndex,
                         float4(previousLambda[0] + appliedDeltaLambda[0],
                                previousLambda[1] + appliedDeltaLambda[1],
                                previousLambda[2] + appliedDeltaLambda[2],
                                0.0f));
        CRESSIM_SB_STORE(g_HingeJointLambdas45, jointIndex,
                         float4(previousLambda[3] + appliedDeltaLambda[3],
                                previousLambda[4] + appliedDeltaLambda[4],
                                lambdaState45.z, lambdaState45.w));
    }
#endif

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
