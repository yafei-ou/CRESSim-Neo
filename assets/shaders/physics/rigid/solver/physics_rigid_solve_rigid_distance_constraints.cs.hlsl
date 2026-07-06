#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

static const float kDistanceConstraintRelaxation = 0.95;
static const float kMaxDistanceConstraintError = 0.08;
static const float kMaxDistanceTranslationCorrection = 0.025;
static const float kMaxDistanceAngularCorrection = 0.15;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRigidDistanceConstraint, g_RigidDistanceConstraints);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_RigidDistanceConstraintLambdas);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint constraintIndex = dispatchThreadID.x;
    if (constraintIndex >= reserved0)
    {
        return;
    }

    const GpuRigidDistanceConstraint constraint =
        CRESSIM_SB_LOAD(g_RigidDistanceConstraints, constraintIndex);
    if (constraint.enabled == 0u)
    {
        CRESSIM_SB_STORE(g_RigidDistanceConstraintLambdas, constraintIndex, 0.0);
        return;
    }
    const uint bodyA = constraint.rigidBodyIndexA;
    const uint bodyB = constraint.rigidBodyIndexB;
    if (bodyA >= rigidBodyCount || bodyB >= rigidBodyCount || bodyA == bodyB)
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

    const float3 rA = QuaternionRotate(qA, constraint.localAnchorA.xyz);
    const float3 rB = QuaternionRotate(qB, constraint.localAnchorB.xyz);
    const float3 pA = posInvMassA.xyz + rA;
    const float3 pB = posInvMassB.xyz + rB;
    const float3 delta = pA - pB;
    const float lengthSq = dot(delta, delta);
    if (lengthSq <= kEpsilon)
    {
        return;
    }

    const float lengthValue = sqrt(lengthSq);
    const float3 gradient = delta / lengthValue;
    const float constraintValue =
        ClampErrorScalar(lengthValue - constraint.restDistance, kMaxDistanceConstraintError);
    if (abs(constraintValue) <= kEpsilon)
    {
        return;
    }

    const float3 angularJacobianA = cross(rA, gradient);
    const float3 angularJacobianB = -cross(rB, gradient);
    float denominator = invMassA + invMassB;
    denominator += dot(angularJacobianA,
                       MultiplyWorldInverseInertia(invInertiaA, qA, angularJacobianA));
    denominator += dot(angularJacobianB,
                       MultiplyWorldInverseInertia(invInertiaB, qB, angularJacobianB));

    const float alpha = constraint.compliance / max(dt * dt, kEpsilon);
    denominator += alpha;
    if (denominator <= kEpsilon)
    {
        return;
    }

    const float lambda = CRESSIM_SB_LOAD(g_RigidDistanceConstraintLambdas, constraintIndex);
    const float deltaLambda = -(constraintValue + alpha * lambda) / denominator;
    CRESSIM_SB_STORE(g_RigidDistanceConstraintLambdas, constraintIndex, lambda + deltaLambda);

    const float3 linearImpulse = gradient * (deltaLambda * kDistanceConstraintRelaxation);
    const float3 angularImpulseA =
        angularJacobianA * (deltaLambda * kDistanceConstraintRelaxation);
    const float3 angularImpulseB =
        angularJacobianB * (deltaLambda * kDistanceConstraintRelaxation);
    const float3 translationA = linearImpulse * invMassA;
    const float3 rotationA = MultiplyWorldInverseInertia(invInertiaA, qA, angularImpulseA);
    const float3 translationB = -linearImpulse * invMassB;
    const float3 rotationB = MultiplyWorldInverseInertia(invInertiaB, qB, angularImpulseB);
    const float correctionScale =
        ComputeCorrectionLimitScale(translationA, rotationA, translationB, rotationB,
                                    kMaxDistanceTranslationCorrection,
                                    kMaxDistanceAngularCorrection);

    if (bodyTypeA == kRigidBodyTypeDynamic && invMassA != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyA,
                                      translationA * correctionScale);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyA,
                                      rotationA * correctionScale);
    }
    if (bodyTypeB == kRigidBodyTypeDynamic && invMassB != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyB,
                                      translationB * correctionScale);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyB,
                                      rotationB * correctionScale);
    }
}
