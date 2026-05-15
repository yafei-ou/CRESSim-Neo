#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/fluid/physics_fluid_anisotropy_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidNeighborCounts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidNeighborPairs);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidAnisotropy1RW);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidAnisotropy2RW);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidAnisotropy3RW);

static const float kFluidRenderAnisotropyScale = 5.0;
static const float kFluidRenderAnisotropyMin = 1.0;
static const float kFluidRenderAnisotropyMax = 2.0;

float3 SafeNormalize(float3 value, float3 fallback)
{
    const float lenSq = dot(value, value);
    return lenSq > 1.0e-8 ? value * rsqrt(lenSq) : fallback;
}

void WriteIsotropicAnisotropy(uint particleIndex, float scale)
{
    CRESSIM_SB_STORE(g_FluidAnisotropy1RW, particleIndex, float4(1.0, 0.0, 0.0, scale));
    CRESSIM_SB_STORE(g_FluidAnisotropy2RW, particleIndex, float4(0.0, 1.0, 0.0, scale));
    CRESSIM_SB_STORE(g_FluidAnisotropy3RW, particleIndex, float4(0.0, 0.0, 1.0, scale));
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
        WriteIsotropicAnisotropy(particleIndex, radius);
        return;
    }

    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float supportRadius = max(fluidMaterial.smoothingRadius, radius);
    const float inverseSupportRadius = rcp(supportRadius);
    const float renderRadius = radius;
    const float isotropicScale = max(kFluidRenderAnisotropyMin * renderRadius, radius);
    const float3 selfPosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const uint neighborCount = CRESSIM_SB_LOAD(g_FluidNeighborCounts, particleIndex);
    const uint neighborOffset = particleIndex * maxFluidNeighborhood;

    float3 weightedMean = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;

    [loop]
    for (uint i = 0u; i < neighborCount; ++i)
    {
        const GpuParticleCandidatePair pair =
            CRESSIM_SB_LOAD(g_FluidNeighborPairs, neighborOffset + i);
        const uint neighborIndex = pair.indexB;
        if (CRESSIM_SB_LOAD(g_ParticleKinds, neighborIndex) != kParticleKindFluid)
        {
            continue;
        }

        const float3 neighborPosition =
            CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, neighborIndex).xyz;
        const float3 delta = selfPosition - neighborPosition;
        const float distanceSq = dot(delta, delta);
        if (distanceSq <= 0.0)
        {
            continue;
        }

        const float distance = sqrt(distanceSq);
        if (distance >= supportRadius)
        {
            continue;
        }

        const float weight = FluidAnisotropyWeight(distance, inverseSupportRadius);
        weightSum += weight;
        weightedMean += neighborPosition * weight;
    }

    if (weightSum <= 0.0)
    {
        WriteIsotropicAnisotropy(particleIndex, isotropicScale);
        return;
    }

    weightedMean /= weightSum;

    float3x3 covariance = float3x3(0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0);

    [loop]
    for (uint i = 0u; i < neighborCount; ++i)
    {
        const GpuParticleCandidatePair pair =
            CRESSIM_SB_LOAD(g_FluidNeighborPairs, neighborOffset + i);
        const uint neighborIndex = pair.indexB;
        if (CRESSIM_SB_LOAD(g_ParticleKinds, neighborIndex) != kParticleKindFluid)
        {
            continue;
        }

        const float3 neighborPosition =
            CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, neighborIndex).xyz;
        const float3 delta = selfPosition - neighborPosition;
        const float distanceSq = dot(delta, delta);
        if (distanceSq <= 0.0)
        {
            continue;
        }

        const float distance = sqrt(distanceSq);
        if (distance >= supportRadius)
        {
            continue;
        }

        const float weight = FluidAnisotropyWeight(distance, inverseSupportRadius);
        const float3 centered = neighborPosition - weightedMean;
        covariance += float3x3(weight * centered.x * centered.x,
                               weight * centered.x * centered.y,
                               weight * centered.x * centered.z,
                               weight * centered.y * centered.x,
                               weight * centered.y * centered.y,
                               weight * centered.y * centered.z,
                               weight * centered.z * centered.x,
                               weight * centered.z * centered.y,
                               weight * centered.z * centered.z);
    }

    covariance /= weightSum;
    covariance[0][1] = covariance[1][0] = 0.5 * (covariance[0][1] + covariance[1][0]);
    covariance[0][2] = covariance[2][0] = 0.5 * (covariance[0][2] + covariance[2][0]);
    covariance[1][2] = covariance[2][1] = 0.5 * (covariance[1][2] + covariance[2][1]);

    float3x3 basis;
    float3 eigenValues;
    FluidAnisotropyEigenDecomposition(covariance, basis, eigenValues);

    float3 axis1 = SafeNormalize(float3(basis[0][0], basis[1][0], basis[2][0]),
                                 float3(1.0, 0.0, 0.0));
    float3 axis2 = SafeNormalize(float3(basis[0][1], basis[1][1], basis[2][1]),
                                 float3(0.0, 1.0, 0.0));
    float3 axis3 = SafeNormalize(float3(basis[0][2], basis[1][2], basis[2][2]),
                                 float3(0.0, 0.0, 1.0));
    FluidAnisotropySortDescending(eigenValues, axis1, axis2, axis3);

    float3 lambda = sqrt(eigenValues);
    lambda *= kFluidRenderAnisotropyScale;
    lambda = FluidAnisotropyClamp(lambda,
                                  kFluidRenderAnisotropyMin * renderRadius,
                                  kFluidRenderAnisotropyMax * renderRadius);

    CRESSIM_SB_STORE(g_FluidAnisotropy1RW, particleIndex, float4(axis1, lambda.x));
    CRESSIM_SB_STORE(g_FluidAnisotropy2RW, particleIndex, float4(axis2, lambda.y));
    CRESSIM_SB_STORE(g_FluidAnisotropy3RW, particleIndex, float4(axis3, lambda.z));
}
