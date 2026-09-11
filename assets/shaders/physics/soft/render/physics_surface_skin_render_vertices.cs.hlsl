#include "physics_surface_render_dispatch_constants.hlsli"
#include "physics_base.hlsli"

struct SurfaceRenderVertexBinding
{
    uint4 particleIndices;
    float4 weights;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(SurfaceRenderVertexBinding, g_SurfaceRenderVertexBindings);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SurfaceRenderPositionsRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint vertexIndex = dispatchThreadID.x;
    if (vertexIndex >= renderVertexCount)
    {
        return;
    }

    const SurfaceRenderVertexBinding binding =
        CRESSIM_SB_LOAD(g_SurfaceRenderVertexBindings, vertexIndex);

    float3 skinnedPos = float3(0.0, 0.0, 0.0);
    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        const uint particleId = binding.particleIndices[i];
        const float weight = binding.weights[i];
        skinnedPos += weight * CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleId).xyz;
    }

    CRESSIM_SB_STORE(g_SurfaceRenderPositionsRW, vertexIndex, float4(skinnedPos, 1.0));
}
