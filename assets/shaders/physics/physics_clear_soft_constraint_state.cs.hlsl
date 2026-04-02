#include "physics/include/physics_soft_dispatch_constants.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftPositionCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftTetLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;

    if (idx < softParticleCount)
    {
        CRESSIM_SB_STORE(g_SoftPositionCorrections, idx, int4(0, 0, 0, 0));
    }

    if (idx < softEdgeCount)
    {
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, idx, 0.0);
    }

    if (idx < softTetCount)
    {
        CRESSIM_SB_STORE(g_SoftTetLambdas, idx, 0.0);
    }
}
