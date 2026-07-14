#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/structured_buffer_compat.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"

static const float kAttachmentRelaxation = 0.95;
static const float kMaxAttachmentTranslationCorrection = 0.05;
static const float kMaxAttachmentAngularCorrection = 0.2;

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRigidParticleAttachmentConstraint, g_RigidParticleAttachments);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidParticleAttachmentLambdas);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);
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

    const GpuRigidParticleAttachmentConstraint constraint =
        CRESSIM_SB_LOAD(g_RigidParticleAttachments, constraintIndex);
    if (constraint.enabled == 0u)
    {
        CRESSIM_SB_STORE(g_RigidParticleAttachmentLambdas, constraintIndex, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }
    if (constraint.particleIndex >= reserved1 || constraint.rigidBodyIndex >= rigidBodyCount)
    {
        return;
    }

    const float4 particlePositionInvMass =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, constraint.particleIndex);
    const float4 rigidPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, constraint.rigidBodyIndex);
    const uint rigidBodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, constraint.rigidBodyIndex);
    const float invMassParticle = particlePositionInvMass.w;
    const float invMassRigid = rigidBodyType == kRigidBodyTypeDynamic ? rigidPositionInvMass.w : 0.0;
    if (invMassParticle <= kEpsilon && invMassRigid <= kEpsilon)
    {
        return;
    }

    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, constraint.rigidBodyIndex));
    float3 invInertiaRigid =
        CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, constraint.rigidBodyIndex).xyz;
    if (invMassRigid <= kEpsilon)
    {
        invInertiaRigid = 0.0;
    }

    const float3 worldLeverArm = QuaternionRotate(rigidOrientation, constraint.localAnchor.xyz);
    const float3 rigidAnchorPosition = rigidPositionInvMass.xyz + worldLeverArm;
    const float3 constraintValue = particlePositionInvMass.xyz - rigidAnchorPosition;
    if (dot(constraintValue, constraintValue) <= kEpsilon * kEpsilon)
    {
        return;
    }

    const float alpha = constraint.compliance / max(dt * dt, kEpsilon);
    const float4 lambdaStorage = CRESSIM_SB_LOAD(g_RigidParticleAttachmentLambdas, constraintIndex);
    float3 lambda = lambdaStorage.xyz;
    float3 particleCorrection = 0.0;
    float3 rigidTranslationCorrection = 0.0;
    float3 rigidRotationCorrection = 0.0;

    [unroll] for (uint axisIndex = 0u; axisIndex < 3u; ++axisIndex)
    {
        const float3 axis = axisIndex == 0u ? float3(1.0, 0.0, 0.0)
                          : axisIndex == 1u ? float3(0.0, 1.0, 0.0)
                                             : float3(0.0, 0.0, 1.0);
        const float c = dot(constraintValue, axis);
        const float3 angularJacobian = cross(worldLeverArm, axis);
        float denominator = invMassParticle + invMassRigid;
        denominator += dot(angularJacobian,
                           MultiplyWorldInverseInertia(invInertiaRigid, rigidOrientation,
                                                       angularJacobian));
        denominator += alpha;
        if (denominator <= kEpsilon)
        {
            continue;
        }

        const float deltaLambda = -(c + alpha * lambda[axisIndex]) / denominator;
        lambda[axisIndex] += deltaLambda;
        particleCorrection += axis * (deltaLambda * invMassParticle * kAttachmentRelaxation);
        rigidTranslationCorrection -= axis * (deltaLambda * invMassRigid * kAttachmentRelaxation);
        rigidRotationCorrection -=
            MultiplyWorldInverseInertia(invInertiaRigid, rigidOrientation, angularJacobian) *
            (deltaLambda * kAttachmentRelaxation);
    }

    CRESSIM_SB_STORE(g_RigidParticleAttachmentLambdas, constraintIndex, float4(lambda, 0.0));

    const float correctionScale = ComputeCorrectionLimitScale(
        particleCorrection, 0.0, rigidTranslationCorrection, rigidRotationCorrection,
        kMaxAttachmentTranslationCorrection, kMaxAttachmentAngularCorrection);

    if (invMassParticle > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, constraint.particleIndex,
                                      particleCorrection * correctionScale);
    }

    if (invMassRigid > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, constraint.rigidBodyIndex,
                                      rigidTranslationCorrection * correctionScale);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, constraint.rigidBodyIndex,
                                      rigidRotationCorrection * correctionScale);
    }
}
