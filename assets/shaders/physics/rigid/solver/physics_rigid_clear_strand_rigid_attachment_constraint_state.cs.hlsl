#include "physics_rigid_dispatch_constants.hlsli"
#include "structured_buffer_compat.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandRigidAttachmentLambda, g_StrandRigidAttachmentLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint constraintIndex = dispatchThreadID.x;
    if (constraintIndex >= reserved0)
    {
        return;
    }

    GpuStrandRigidAttachmentLambda lambdaState;
    lambdaState.translation = 0.0;
    lambdaState.rotation    = 0.0;
    CRESSIM_SB_STORE(g_StrandRigidAttachmentLambdas, constraintIndex, lambdaState);
}
