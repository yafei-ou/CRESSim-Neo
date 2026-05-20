#include "../../../include/physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "../../../include/structured_buffer_compat.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_HingeJointLambdas0123);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_HingeJointLambdas45);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint jointIndex = dispatchThreadID.x;
    if (jointIndex >= jointCount)
    {
        return;
    }

    CRESSIM_SB_STORE(g_HingeJointLambdas0123, jointIndex, 0.0);
    CRESSIM_SB_STORE(g_HingeJointLambdas45, jointIndex, 0.0);
}
