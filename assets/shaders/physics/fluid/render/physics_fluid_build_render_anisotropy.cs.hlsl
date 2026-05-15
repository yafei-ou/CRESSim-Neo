#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidSurfaceNormalConstraints);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidAnisotropy1RW);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidAnisotropy2RW);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidAnisotropy3RW);

float3 safeNormalize(float3 value, float3 fallback)
{
    const float lenSq = dot(value, value);
    return lenSq > 1.0e-8 ? value * rsqrt(lenSq) : fallback;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    const float radius = max(CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex), 1.0e-4);
    if (CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        CRESSIM_SB_STORE(g_FluidAnisotropy1RW, particleIndex, float4(1.0, 0.0, 0.0, radius));
        CRESSIM_SB_STORE(g_FluidAnisotropy2RW, particleIndex, float4(0.0, 1.0, 0.0, radius));
        CRESSIM_SB_STORE(g_FluidAnisotropy3RW, particleIndex, float4(0.0, 0.0, 1.0, radius));
        return;
    }

    const float4 normalConstraint = CRESSIM_SB_LOAD(g_FluidSurfaceNormalConstraints, particleIndex);
    const float constraint = saturate(abs(normalConstraint.w));
    const float3 normal = safeNormalize(normalConstraint.xyz, float3(0.0, 1.0, 0.0));
    const float3 refAxis = abs(normal.y) < 0.95 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = safeNormalize(cross(refAxis, normal), float3(1.0, 0.0, 0.0));
    const float3 bitangent = safeNormalize(cross(normal, tangent), float3(0.0, 0.0, 1.0));

    const float lateralScale = radius * lerp(1.0, 1.75, constraint);
    const float normalScale = radius * lerp(1.0, 0.55, constraint);

    CRESSIM_SB_STORE(g_FluidAnisotropy1RW, particleIndex, float4(tangent, lateralScale));
    CRESSIM_SB_STORE(g_FluidAnisotropy2RW, particleIndex, float4(bitangent, lateralScale));
    CRESSIM_SB_STORE(g_FluidAnisotropy3RW, particleIndex, float4(normal, normalScale));
}
