#include "physics/physics_rigid_dispatch_constants.hlsli"
#include "physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/rigid/physics_rigid_joint_solver_shared.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(GpuHingeJoint, g_HingeJoints);
CRESSIM_RW_STRUCTURED_BUFFER(GpuHingeJointRuntimeState, g_HingeJointRuntimeStates);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint jointIndex = dispatchThreadID.x;
    if (jointIndex >= jointCount)
    {
        return;
    }

    const GpuHingeJoint joint = CRESSIM_SB_LOAD(g_HingeJoints, jointIndex);
    if (joint.bodyA >= rigidBodyCount || joint.bodyB >= rigidBodyCount)
    {
        return;
    }

    const float4 qA =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, joint.bodyA));
    const float4 qB =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, joint.bodyB));
    const float3 angularVelocityA =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, joint.bodyA).xyz;
    const float3 angularVelocityB =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, joint.bodyB).xyz;
    const float currentWrappedAngle =
        ComputeHingeWrappedAngle(qA, qB, joint.localAxisA0.xyz, joint.localAxisA1.xyz,
                                 joint.localAxisB1.xyz);
    const float3 hingeAxis =
        SafeNormalize(QuaternionRotate(qA, joint.localAxisA0.xyz), float3(1.0, 0.0, 0.0));
    const float currentAngularVelocity = dot(angularVelocityB - angularVelocityA, hingeAxis);

    GpuHingeJointRuntimeState state = CRESSIM_SB_LOAD(g_HingeJointRuntimeStates, jointIndex);
    const bool initialized = state.angleState.w > 0.5;
    const float currentUnwrappedAngle =
        initialized ? state.angleState.y + WrapAngleDelta(currentWrappedAngle - state.angleState.x)
                    : currentWrappedAngle;
    state.angleState = float4(currentWrappedAngle, currentUnwrappedAngle, currentAngularVelocity, 1.0);
    CRESSIM_SB_STORE(g_HingeJointRuntimeStates, jointIndex, state);
}
