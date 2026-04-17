#include "physics/include/physics_soft_render_dispatch_constants.hlsli"

struct GpuSoftRenderVertexTriangleRange
{
    uint start;
    uint count;
    uint reserved0;
    uint reserved1;
};

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftRenderTriangleNormals);
CRESSIM_STRUCTURED_BUFFER(GpuSoftRenderVertexTriangleRange, g_SoftRenderVertexTriangleRanges);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftRenderVertexTriangleIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftRenderFallbackNormals);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftBodyRenderNormalsRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint vertexIndex = dispatchThreadID.x;
    if (vertexIndex >= renderVertexCount)
    {
        return;
    }

    const GpuSoftRenderVertexTriangleRange range =
        CRESSIM_SB_LOAD(g_SoftRenderVertexTriangleRanges, vertexIndex);
    float3 accumulated = float3(0.0, 0.0, 0.0);
    for (uint i = 0u; i < range.count; ++i)
    {
        const uint triangleIndex = CRESSIM_SB_LOAD(g_SoftRenderVertexTriangleIndices, range.start + i);
        accumulated += CRESSIM_SB_LOAD(g_SoftRenderTriangleNormals, triangleIndex).xyz;
    }

    float3 normal = accumulated;
    const float lenSq = dot(normal, normal);
    if (lenSq <= 1.0e-12)
    {
        normal = CRESSIM_SB_LOAD(g_SoftRenderFallbackNormals, vertexIndex).xyz;
    }
    else
    {
        normal *= rsqrt(lenSq);
    }

    CRESSIM_SB_STORE(g_SoftBodyRenderNormalsRW, vertexIndex, float4(normal, 0.0));
}
