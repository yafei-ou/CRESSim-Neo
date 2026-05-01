#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

static const float kJointRelaxation = 0.95;
static const float kMaxJointError = 0.05;

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

    const float3 a0 = SafeNormalize(QuaternionRotate(qA, joint.localAxisA0.xyz), float3(1.0, 0.0, 0.0));
    const float3 a1 = SafeNormalize(QuaternionRotate(qA, joint.localAxisA1.xyz), ChoosePerpendicular(a0));
    const float3 a2 = SafeNormalize(QuaternionRotate(qA, joint.localAxisA2.xyz), normalize(cross(a0, a1)));
    const float3 b0 = SafeNormalize(QuaternionRotate(qB, joint.localAxisB0.xyz), a0);
    const float3 b1 = SafeNormalize(QuaternionRotate(qB, joint.localAxisB1.xyz), a1);
    const float3 b2 = SafeNormalize(QuaternionRotate(qB, joint.localAxisB2.xyz), a2);

    const float3 delta = clamp(posInvMassA.xyz - posInvMassB.xyz, -kMaxJointError, kMaxJointError);

    float3 translationA = 0.0;
    float3 rotationA = 0.0;
    float3 translationB = 0.0;
    float3 rotationB = 0.0;

    float3 rowTranslationA;
    float3 rowRotationA;
    float3 rowTranslationB;
    float3 rowRotationB;
    ApplyLinearConstraintRow(
        a1, -(dot(delta, a1) - joint.referenceOffset.x) * kJointRelaxation, invMassA, invMassB,
        invInertiaA, qA, float3(0.0, 0.0, 0.0), invInertiaB, qB, float3(0.0, 0.0, 0.0),
        rowTranslationA, rowRotationA, rowTranslationB, rowRotationB);
    translationA += rowTranslationA;
    rotationA += rowRotationA;
    translationB += rowTranslationB;
    rotationB += rowRotationB;

    ApplyLinearConstraintRow(
        a2, -(dot(delta, a2) - joint.referenceOffset.y) * kJointRelaxation, invMassA, invMassB,
        invInertiaA, qA, float3(0.0, 0.0, 0.0), invInertiaB, qB, float3(0.0, 0.0, 0.0),
        rowTranslationA, rowRotationA, rowTranslationB, rowRotationB);
    translationA += rowTranslationA;
    rotationA += rowRotationA;
    translationB += rowTranslationB;
    rotationB += rowRotationB;

    const float3 angularError = 0.5 * (cross(b0, a0) + cross(b1, a1) + cross(b2, a2));

    ApplyAngularConstraintRow(a0, dot(angularError, a0) * kJointRelaxation,
                              invInertiaA, qA, invInertiaB, qB, rowRotationA, rowRotationB);
    rotationA += rowRotationA;
    rotationB += rowRotationB;

    ApplyAngularConstraintRow(a1, dot(angularError, a1) * kJointRelaxation,
                              invInertiaA, qA, invInertiaB, qB, rowRotationA, rowRotationB);
    rotationA += rowRotationA;
    rotationB += rowRotationB;

    ApplyAngularConstraintRow(a2, dot(angularError, a2) * kJointRelaxation,
                              invInertiaA, qA, invInertiaB, qB, rowRotationA, rowRotationB);
    rotationA += rowRotationA;
    rotationB += rowRotationB;

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
