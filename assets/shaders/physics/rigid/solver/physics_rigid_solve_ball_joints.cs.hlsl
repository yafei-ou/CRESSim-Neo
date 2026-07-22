#include "physics_rigid_dispatch_constants.hlsli"
#include "physics_rigid_joint_dispatch_constants.hlsli"
#include "physics_atomic_float.hlsli"
#include "physics_rigid_types.hlsli"
#include "physics_rigid_joint_solver_shared.hlsli"


CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuBallJoint, g_BallJoints);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint jointIndex = dispatchThreadID.x;
    if (jointIndex >= jointCount)
    {
        return;
    }

    const GpuBallJoint joint = CRESSIM_SB_LOAD(g_BallJoints, jointIndex);
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
    const float3 delta = ClampErrorVector(pA - pB, kMaxJointError) * kBallJointRelaxation;

    const float3 n0 = float3(1.0, 0.0, 0.0);
    const float3 n1 = float3(0.0, 1.0, 0.0);
    const float3 n2 = float3(0.0, 0.0, 1.0);

    const float3 jAngA0 = cross(rA, n0);
    const float3 jAngA1 = cross(rA, n1);
    const float3 jAngA2 = cross(rA, n2);
    const float3 jAngB0 = -cross(rB, n0);
    const float3 jAngB1 = -cross(rB, n1);
    const float3 jAngB2 = -cross(rB, n2);

    const float k00 = ComputeConstraintMatrixElement(invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                                                     n0, jAngA0, -n0, jAngB0,
                                                     n0, jAngA0, -n0, jAngB0);
    const float k01 = ComputeConstraintMatrixElement(invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                                                     n0, jAngA0, -n0, jAngB0,
                                                     n1, jAngA1, -n1, jAngB1);
    const float k02 = ComputeConstraintMatrixElement(invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                                                     n0, jAngA0, -n0, jAngB0,
                                                     n2, jAngA2, -n2, jAngB2);
    const float k10 = k01;
    const float k11 = ComputeConstraintMatrixElement(invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                                                     n1, jAngA1, -n1, jAngB1,
                                                     n1, jAngA1, -n1, jAngB1);
    const float k12 = ComputeConstraintMatrixElement(invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                                                     n1, jAngA1, -n1, jAngB1,
                                                     n2, jAngA2, -n2, jAngB2);
    const float k20 = k02;
    const float k21 = k12;
    const float k22 = ComputeConstraintMatrixElement(invMassA, invInertiaA, qA, invMassB, invInertiaB, qB,
                                                     n2, jAngA2, -n2, jAngB2,
                                                     n2, jAngA2, -n2, jAngB2);

    float3 lambda = 0.0;
    if (!SolveLinearSystem3x3(k00, k01, k02,
                              k10, k11, k12,
                              k20, k21, k22,
                              -delta.x, -delta.y, -delta.z, lambda))
    {
        return;
    }

    const float3 linearImpulse = n0 * lambda.x + n1 * lambda.y + n2 * lambda.z;
    const float3 angularImpulseA = jAngA0 * lambda.x + jAngA1 * lambda.y + jAngA2 * lambda.z;
    const float3 angularImpulseB = jAngB0 * lambda.x + jAngB1 * lambda.y + jAngB2 * lambda.z;

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
