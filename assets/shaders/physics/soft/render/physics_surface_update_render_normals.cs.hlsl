#include "physics_surface_render_dispatch_constants.hlsli"
#include "physics_base.hlsli"

struct GpuSurfaceRenderVertexTriangleRange
{
    uint start;
    uint count;
    uint reserved0;
    uint reserved1;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_SurfaceRenderTriangleNormals);
CRESSIM_STRUCTURED_BUFFER(GpuSurfaceRenderVertexTriangleRange, g_SurfaceRenderVertexTriangleRanges);
CRESSIM_STRUCTURED_BUFFER(uint, g_SurfaceRenderVertexTriangleIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_SurfaceRenderFallbackNormals);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SurfaceRenderNormalsRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint vertexIndex = dispatchThreadID.x;
    if (vertexIndex >= renderVertexCount)
    {
        return;
    }

    const GpuSurfaceRenderVertexTriangleRange range =
        CRESSIM_SB_LOAD(g_SurfaceRenderVertexTriangleRanges, vertexIndex);
    float3 accumulated = float3(0.0, 0.0, 0.0);
    for (uint i = 0u; i < range.count; ++i)
    {
        const uint triangleIndex = CRESSIM_SB_LOAD(g_SurfaceRenderVertexTriangleIndices, range.start + i);
        accumulated += CRESSIM_SB_LOAD(g_SurfaceRenderTriangleNormals, triangleIndex).xyz;
    }

    float3 normal = accumulated;
    const float lenSq = dot(normal, normal);
    if (lenSq <= 1.0e-12)
    {
        normal = CRESSIM_SB_LOAD(g_SurfaceRenderFallbackNormals, vertexIndex).xyz;
    }
    else
    {
        normal *= rsqrt(lenSq);
    }

    CRESSIM_SB_STORE(g_SurfaceRenderNormalsRW, vertexIndex, float4(normal, 0.0));
}
