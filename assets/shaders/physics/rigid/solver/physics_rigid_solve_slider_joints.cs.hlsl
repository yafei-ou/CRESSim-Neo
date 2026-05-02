#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

static const float kJointRelaxation = 0.95;
static const float kMaxJointError = 0.05;
static const float kSliderTranslationRegularization = 1e-5;
static const float kSliderAngularRegularization = 5e-5;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);

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
    const float3 delta = ClampErrorVector(pA - pB, kMaxJointError);
    const float3 tRow0 = ComputeProjectionJacobianRow(joint.projectionRow0, qA, qB);
    const float3 tRow1 = ComputeProjectionJacobianRow(joint.projectionRow1, qA, qB);
    const float3 tRow2 = ComputeProjectionJacobianRow(joint.projectionRow2, qA, qB);

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

    float rhs[5];
    rhs[0] = -dot(delta, a1) * kJointRelaxation;
    rhs[1] = -dot(delta, a2) * kJointRelaxation;
    rhs[2] = -ComputeProjectionConstraintValue(joint.projectionRow0, qA, qB) * kJointRelaxation;
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

    // Regularize the slider system to reduce sensitivity in near-symmetric
    // frame configurations while preserving the fully coupled 5x5 structure.
    k[0][0] += kSliderTranslationRegularization;
    k[1][1] += kSliderTranslationRegularization;
    k[2][2] += kSliderAngularRegularization;
    k[3][3] += kSliderAngularRegularization;
    k[4][4] += kSliderAngularRegularization;

    float lambda[5];
    if (!SolveLinearSystem5x5(k, rhs, lambda))
    {
        return;
    }

    float3 linearImpulse = 0.0;
    float3 angularImpulseA = 0.0;
    float3 angularImpulseB = 0.0;
    [unroll] for (uint row = 0u; row < 5u; ++row)
    {
        linearImpulse += jLinA[row] * lambda[row];
        angularImpulseA += jAngA[row] * lambda[row];
        angularImpulseB += jAngB[row] * lambda[row];
    }

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
