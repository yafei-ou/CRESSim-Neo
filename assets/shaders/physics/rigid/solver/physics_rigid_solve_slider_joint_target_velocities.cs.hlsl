#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

static const float kVelocityRegularization = 1e-5;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);
CRESSIM_STRUCTURED_BUFFER(uint, g_SliderJointIndices);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyAngularVelocityCorrections);

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

    const float3 linearVelocityA = CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, bodyA).xyz;
    const float3 linearVelocityB = CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, bodyB).xyz;
    const float3 angularVelocityA = CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, bodyA).xyz;
    const float3 angularVelocityB = CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, bodyB).xyz;

    const float3 axis0 =
        SafeNormalize(QuaternionRotate(qA, joint.localAxisA0.xyz), float3(1.0, 0.0, 0.0));
    const float3 linearConstraint =
        (linearVelocityA - linearVelocityB) + joint.driveTargetParams.z * axis0;
    const float3 angularConstraint = angularVelocityA - angularVelocityB;

    const float3 basis[3] = {
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0),
    };

    float3 jLinA[6];
    float3 jAngA[6];
    float3 jLinB[6];
    float3 jAngB[6];
    [unroll] for (uint i = 0u; i < 3u; ++i)
    {
        jLinA[i] = basis[i];
        jAngA[i] = 0.0;
        jLinB[i] = -basis[i];
        jAngB[i] = 0.0;
        jLinA[3u + i] = 0.0;
        jAngA[3u + i] = basis[i];
        jLinB[3u + i] = 0.0;
        jAngB[3u + i] = -basis[i];
    }

    float rhs[6];
    rhs[0] = -linearConstraint.x;
    rhs[1] = -linearConstraint.y;
    rhs[2] = -linearConstraint.z;
    rhs[3] = -angularConstraint.x;
    rhs[4] = -angularConstraint.y;
    rhs[5] = -angularConstraint.z;

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
        k[row][row] += kVelocityRegularization;
    }

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

    if (bodyTypeA == kRigidBodyTypeDynamic && invMassA != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, bodyA,
                                      linearImpulse * invMassA);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyAngularVelocityCorrections, bodyA,
                                      MultiplyWorldInverseInertia(invInertiaA, qA, angularImpulseA));
    }
    if (bodyTypeB == kRigidBodyTypeDynamic && invMassB != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, bodyB,
                                      -linearImpulse * invMassB);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyAngularVelocityCorrections, bodyB,
                                      MultiplyWorldInverseInertia(invInertiaB, qB, angularImpulseB));
    }
}
