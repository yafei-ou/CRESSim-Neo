#include "physics/physics_rigid_joint_dispatch_constants.hlsli"
#include "structured_buffer_compat.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SphericalJointTranslationLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SphericalJointRotationLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint jointIndex = dispatchThreadID.x;
    if (jointIndex >= jointCount)
    {
        return;
    }

    CRESSIM_SB_STORE(g_SphericalJointTranslationLambdas, jointIndex, 0.0);
    CRESSIM_SB_STORE(g_SphericalJointRotationLambdas, jointIndex, 0.0);
}
