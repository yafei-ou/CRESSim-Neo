#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

static const float kJointRelaxation = 0.7;
static const float kMaxJointError = 0.05;
static const float kMaxJointTranslationCorrection = 0.02;
static const float kMaxJointAngularCorrection = 0.12;
static const float kJointRegularization = 1e-5;
static const float kMinXpbdDt = 1e-5;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuSphericalJoint, g_SphericalJoints);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SphericalJointTranslationLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SphericalJointRotationLambdas);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

float4 EnsureHemisphere(float4 q)
{
    return q.w < 0.0 ? -q : q;
}

float3 QuaternionToRotationVector(float4 q)
{
    const float4 normalized = EnsureHemisphere(QuaternionNormalize(q));
    return 2.0 * normalized.xyz;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint jointIndex = dispatchThreadID.x;
    if (jointIndex >= jointCount)
    {
        return;
    }

    const GpuSphericalJoint joint = CRESSIM_SB_LOAD(g_SphericalJoints, jointIndex);
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

    const float invDtSq = 1.0 / max(dt * dt, kMinXpbdDt * kMinXpbdDt);

    float3 linearImpulse = 0.0;
    float3 angularImpulseA = 0.0;
    float3 angularImpulseB = 0.0;

    const float3 rA = QuaternionRotate(qA, joint.localAnchorA.xyz);
    const float3 rB = QuaternionRotate(qB, joint.localAnchorB.xyz);
    const float3 pA = posInvMassA.xyz + rA;
    const float3 pB = posInvMassB.xyz + rB;
    const float3 delta = ClampErrorVector(pA - pB, kMaxJointError);
    float3 translationLambda = CRESSIM_SB_LOAD(g_SphericalJointTranslationLambdas, jointIndex).xyz;
    const float translationAlpha = max(joint.limitParams0.w, 0.0) * invDtSq;

    [unroll] for (uint axisIndex = 0u; axisIndex < 3u; ++axisIndex)
    {
        const float3 axis = axisIndex == 0u ? float3(1.0, 0.0, 0.0)
                          : axisIndex == 1u ? float3(0.0, 1.0, 0.0)
                                             : float3(0.0, 0.0, 1.0);
        const float3 jAngA = cross(rA, axis);
        const float3 jAngB = -cross(rB, axis);
        float denominator = invMassA + invMassB;
        denominator += dot(jAngA, MultiplyWorldInverseInertia(invInertiaA, qA, jAngA));
        denominator += dot(jAngB, MultiplyWorldInverseInertia(invInertiaB, qB, jAngB));
        denominator += kJointRegularization + translationAlpha;
        if (denominator <= kEpsilon)
        {
            continue;
        }

        const float error = delta[axisIndex];
        const float deltaLambda =
            (-(error + translationAlpha * translationLambda[axisIndex]) / denominator) *
            kJointRelaxation;
        translationLambda[axisIndex] += deltaLambda;
        linearImpulse += axis * deltaLambda;
        angularImpulseA += jAngA * deltaLambda;
        angularImpulseB += jAngB * deltaLambda;
    }
    CRESSIM_SB_STORE(g_SphericalJointTranslationLambdas, jointIndex, float4(translationLambda, 0.0));

    const float4 frameA = QuaternionNormalize(QuaternionMul(qA, joint.localRotationA));
    const float4 frameB = QuaternionNormalize(QuaternionMul(qB, joint.localRotationB));
    const float4 relative = QuaternionNormalize(QuaternionMul(QuaternionConjugate(frameA), frameB));
    float3 targetAngles = 0.0;
    if (joint.driveMode == kRigidJointDriveModeTargetOrientation)
    {
        targetAngles = QuaternionToRotationVector(joint.driveTargetOrientation);
    }
    if (joint.limitParams0.x > 0.5)
    {
        targetAngles.x = clamp(targetAngles.x, joint.limitParams1.x, joint.limitParams1.y);
        targetAngles.y = clamp(targetAngles.y, -joint.limitParams0.y, joint.limitParams0.y);
        targetAngles.z = clamp(targetAngles.z, -joint.limitParams0.z, joint.limitParams0.z);
    }
    const float3 currentAngles = QuaternionToRotationVector(relative);
    const float3 rotationError = currentAngles - targetAngles;
    float3 rotationLambda = CRESSIM_SB_LOAD(g_SphericalJointRotationLambdas, jointIndex).xyz;
    const float baseDriveCompliance =
        joint.driveMode == kRigidJointDriveModeTargetOrientation ? joint.driveParams.x : 0.0;
    const float3 localAxes[3] = {float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0),
                                 float3(0.0, 0.0, 1.0)};

    [unroll] for (uint axisIndex = 0u; axisIndex < 3u; ++axisIndex)
    {
        const float3 worldAxis = QuaternionRotate(frameA, localAxes[axisIndex]);
        const float compliance = axisIndex == 0u ? joint.limitParams1.w : joint.limitParams1.z;
        const float alpha = max(compliance + baseDriveCompliance, 0.0) * invDtSq;
        float denominator = dot(worldAxis, MultiplyWorldInverseInertia(invInertiaA, qA, worldAxis));
        denominator += dot(worldAxis, MultiplyWorldInverseInertia(invInertiaB, qB, worldAxis));
        denominator += kJointRegularization + alpha;
        if (denominator <= kEpsilon)
        {
            continue;
        }

        const float deltaLambda =
            (-(rotationError[axisIndex] + alpha * rotationLambda[axisIndex]) / denominator) *
            kJointRelaxation;
        rotationLambda[axisIndex] += deltaLambda;
        angularImpulseA += -worldAxis * deltaLambda;
        angularImpulseB += worldAxis * deltaLambda;
    }
    CRESSIM_SB_STORE(g_SphericalJointRotationLambdas, jointIndex, float4(rotationLambda, 0.0));

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
