#include "physics_surface_render_dispatch_constants.hlsli"
#include "physics_base.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SurfaceRenderTriangleParticleIndices);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SurfaceRenderTriangleNormalsRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint triangleIndex = dispatchThreadID.x;
    if (triangleIndex >= renderTriangleCount)
    {
        return;
    }

    const uint4 triangleIndices =
        CRESSIM_SB_LOAD(g_SurfaceRenderTriangleParticleIndices, triangleIndex);
    const float3 p0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, triangleIndices.x).xyz;
    const float3 p1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, triangleIndices.y).xyz;
    const float3 p2 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, triangleIndices.z).xyz;
    const float3 faceNormal = cross(p2 - p0, p1 - p0);
    CRESSIM_SB_STORE(g_SurfaceRenderTriangleNormalsRW, triangleIndex, float4(faceNormal, 0.0));
}
