#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_joint_solver_shared.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(GpuSliderJoint, g_SliderJoints);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSliderJointRuntimeState, g_SliderJointRuntimeStates);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint jointIndex = dispatchThreadID.x;
    if (jointIndex >= jointCount)
    {
        return;
    }

    const GpuSliderJoint joint = CRESSIM_SB_LOAD(g_SliderJoints, jointIndex);
    if (joint.bodyA >= rigidBodyCount || joint.bodyB >= rigidBodyCount)
    {
        return;
    }

    const float4 posInvMassA =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, joint.bodyA);
    const float4 posInvMassB =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, joint.bodyB);
    const float4 qA =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, joint.bodyA));
    const float4 qB =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, joint.bodyB));
    const float3 linearVelocityA =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, joint.bodyA).xyz;
    const float3 linearVelocityB =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, joint.bodyB).xyz;
    const float3 angularVelocityA =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, joint.bodyA).xyz;
    const float3 angularVelocityB =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, joint.bodyB).xyz;

    const float3 axis0 =
        SafeNormalize(QuaternionRotate(qA, joint.localAxisA0.xyz), float3(1.0, 0.0, 0.0));
    const float3 rA = QuaternionRotate(qA, joint.localAnchorA.xyz);
    const float3 rB = QuaternionRotate(qB, joint.localAnchorB.xyz);
    const float currentPosition =
        joint.driveTargetParams.y - dot((posInvMassA.xyz + rA) - (posInvMassB.xyz + rB), axis0);
    const float3 anchorVelocityA = linearVelocityA + cross(angularVelocityA, rA);
    const float3 anchorVelocityB = linearVelocityB + cross(angularVelocityB, rB);
    const float currentVelocity = dot(anchorVelocityB - anchorVelocityA, axis0);

    GpuSliderJointRuntimeState state{};
    state.state = float4(currentPosition, currentVelocity, 0.0, 0.0);
    CRESSIM_SB_STORE(g_SliderJointRuntimeStates, jointIndex, state);
}
