#include "physics/physics_rigid_dispatch_constants.hlsli"
#include "structured_buffer_compat.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_RigidParticleAttachmentLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint constraintIndex = dispatchThreadID.x;
    if (constraintIndex >= reserved0)
    {
        return;
    }

    CRESSIM_SB_STORE(g_RigidParticleAttachmentLambdas, constraintIndex, float4(0.0, 0.0, 0.0, 0.0));
}
