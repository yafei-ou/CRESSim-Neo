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
CRESSIM_STRUCTURED_BUFFER(GpuHingeJoint, g_HingeJoints);

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
    const float3 delta = clamp(pA - pB, -kMaxJointError, kMaxJointError);

    const float3 anchorBasis[3] = {float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0)};

    float3 translationA = 0.0;
    float3 rotationA = 0.0;
    float3 translationB = 0.0;
    float3 rotationB = 0.0;

    [unroll] for (uint row = 0u; row < 3u; ++row)
    {
        float3 rowTranslationA;
        float3 rowRotationA;
        float3 rowTranslationB;
        float3 rowRotationB;
        ApplyLinearConstraintRow(
            anchorBasis[row], -dot(delta, anchorBasis[row]) * kJointRelaxation, invMassA, invMassB,
            invInertiaA, qA, rA, invInertiaB, qB, rB,
            rowTranslationA, rowRotationA, rowTranslationB, rowRotationB);
        translationA += rowTranslationA;
        rotationA += rowRotationA;
        translationB += rowTranslationB;
        rotationB += rowRotationB;
    }

    const float3 axisA = SafeNormalize(QuaternionRotate(qA, joint.localAxisA.xyz), float3(1.0, 0.0, 0.0));
    const float3 axisB = SafeNormalize(QuaternionRotate(qB, joint.localAxisB.xyz), axisA);
    const float3 angularError = cross(axisB, axisA);
    const float3 row0 = QuaternionRotate(qA, joint.projectionRow0.xyz);
    const float3 row1 = QuaternionRotate(qA, joint.projectionRow1.xyz);

    float3 rowRotationA;
    float3 rowRotationB;
    ApplyAngularConstraintRow(row0, dot(angularError, row0) * kJointRelaxation,
                              invInertiaA, qA, invInertiaB, qB, rowRotationA, rowRotationB);
    rotationA += rowRotationA;
    rotationB += rowRotationB;

    ApplyAngularConstraintRow(row1, dot(angularError, row1) * kJointRelaxation,
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
