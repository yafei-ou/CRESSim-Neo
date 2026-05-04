#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

static const float kJointRelaxation = 0.95;
static const float kMaxJointError = 0.05;
static const float kJointDriveRelaxation = 0.2;
static const float kHingeTranslationRegularization = 1e-5;
static const float kHingeAngularRegularization = 5e-5;

#ifndef CRESSIM_JOINT_DRIVE_MODE_TARGET_POSITION
#define CRESSIM_JOINT_DRIVE_MODE_TARGET_POSITION 0
#endif

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuHingeJoint, g_HingeJoints);
CRESSIM_STRUCTURED_BUFFER(uint, g_HingeJointIndices);

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
    const float3 delta = ClampErrorVector(pA - pB, kMaxJointError) * kJointRelaxation;

    const float3 tRow0 = ComputeProjectionJacobianRow(joint.projectionRow0, qA, qB);
    const float3 tRow1 = ComputeProjectionJacobianRow(joint.projectionRow1, qA, qB);
    const float3 tRow2 = ComputeProjectionJacobianRow(joint.projectionRow2, qA, qB);
    const bool limitEnabled = joint.limitParams.x > 0.5;
    const float2 limitRange = joint.limitParams.yz;
    const float currentAngle = ComputeHingeAngle(joint.projectionRow0, qA, qB);
    float limitTargetAngle = currentAngle;
    const bool limitActive = limitEnabled && ComputeLimitTarget(currentAngle, limitRange, limitTargetAngle);

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

    float rhs[6];
    rhs[0] = -delta.x;
    rhs[1] = -delta.y;
    rhs[2] = -delta.z;
    const float driveTargetAngle =
        limitEnabled ? clamp(joint.driveTargetParams.x, limitRange.x, limitRange.y)
                     : joint.driveTargetParams.x;
    rhs[3] = -(ComputeProjectionConstraintValue(joint.projectionRow0, qA, qB) -
               sin(0.5 * driveTargetAngle)) * kJointDriveRelaxation;
    rhs[4] = -ComputeProjectionConstraintValue(joint.projectionRow1, qA, qB) * kJointRelaxation;
    rhs[5] = -ComputeProjectionConstraintValue(joint.projectionRow2, qA, qB) * kJointRelaxation;

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

    k[0][0] += kHingeTranslationRegularization;
    k[1][1] += kHingeTranslationRegularization;
    k[2][2] += kHingeTranslationRegularization;
    k[3][3] += kHingeAngularRegularization;
    k[4][4] += kHingeAngularRegularization;
    k[5][5] += kHingeAngularRegularization;

    float lambda[6];
    if (!SolveLinearSystem6x6(k, rhs, lambda))
    {
        return;
    }

    float3 linearImpulse = 0.0;
    float3 angularImpulseA = 0.0;
    float3 angularImpulseB = 0.0;
    [unroll] for (uint row = 0u; row < 6u; ++row)
    {
        linearImpulse += jLinA[row] * lambda[row];
        angularImpulseA += jAngA[row] * lambda[row];
        angularImpulseB += jAngB[row] * lambda[row];
    }
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

        float rhs[6];
        rhs[0] = -delta.x;
        rhs[1] = -delta.y;
        rhs[2] = -delta.z;
        rhs[3] = -(ComputeProjectionConstraintValue(joint.projectionRow0, qA, qB) -
                   sin(0.5 * limitTargetAngle)) * kJointRelaxation;
        rhs[4] = -ComputeProjectionConstraintValue(joint.projectionRow1, qA, qB) * kJointRelaxation;
        rhs[5] = -ComputeProjectionConstraintValue(joint.projectionRow2, qA, qB) * kJointRelaxation;

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

        k[0][0] += kHingeTranslationRegularization;
        k[1][1] += kHingeTranslationRegularization;
        k[2][2] += kHingeTranslationRegularization;
        k[3][3] += kHingeAngularRegularization;
        k[4][4] += kHingeAngularRegularization;
        k[5][5] += kHingeAngularRegularization;

        float lambda[6];
        if (!SolveLinearSystem6x6(k, rhs, lambda))
        {
            return;
        }

        [unroll] for (uint row = 0u; row < 6u; ++row)
        {
            linearImpulse += jLinA[row] * lambda[row];
            angularImpulseA += jAngA[row] * lambda[row];
            angularImpulseB += jAngB[row] * lambda[row];
        }
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

        float rhs[5];
        rhs[0] = -delta.x;
        rhs[1] = -delta.y;
        rhs[2] = -delta.z;
        rhs[3] = -ComputeProjectionConstraintValue(joint.projectionRow1, qA, qB) * kJointRelaxation;
        rhs[4] = -ComputeProjectionConstraintValue(joint.projectionRow2, qA, qB) * kJointRelaxation;

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

        float lambda[5];
        if (!SolveLinearSystem5x5(k, rhs, lambda))
        {
            return;
        }

        [unroll] for (uint row = 0u; row < 5u; ++row)
        {
            linearImpulse += jLinA[row] * lambda[row];
            angularImpulseA += jAngA[row] * lambda[row];
            angularImpulseB += jAngB[row] * lambda[row];
        }
    }
#endif

    const float3 translationA = linearImpulse * invMassA;
    const float3 rotationA = MultiplyWorldInverseInertia(invInertiaA, qA, angularImpulseA);
    const float3 translationB = -linearImpulse * invMassB;
    const float3 rotationB = MultiplyWorldInverseInertia(invInertiaB, qB, angularImpulseB);

    if (bodyTypeA == kRigidBodyTypeDynamic && invMassA != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyA, translationA);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyA, rotationA);
    }
    if (bodyTypeB == kRigidBodyTypeDynamic && invMassB != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyB, translationB);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyB, rotationB);
    }
}
