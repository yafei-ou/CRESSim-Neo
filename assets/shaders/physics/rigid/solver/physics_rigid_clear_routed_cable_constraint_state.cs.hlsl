#include "physics/physics_rigid_dispatch_constants.hlsli"
#include "structured_buffer_compat.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float, g_RoutedCableLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint cableIndex = dispatchThreadID.x;
    if (cableIndex >= reserved0)
    {
        return;
    }

    CRESSIM_SB_STORE(g_RoutedCableLambdas, cableIndex, 0.0);
}
