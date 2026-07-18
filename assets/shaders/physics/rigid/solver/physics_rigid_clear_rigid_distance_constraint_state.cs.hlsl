#include "physics_rigid_dispatch_constants.hlsli"
#include "structured_buffer_compat.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float, g_RigidDistanceConstraintLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint constraintIndex = dispatchThreadID.x;
    if (constraintIndex >= reserved0)
    {
        return;
    }

    CRESSIM_SB_STORE(g_RigidDistanceConstraintLambdas, constraintIndex, 0.0);
}
