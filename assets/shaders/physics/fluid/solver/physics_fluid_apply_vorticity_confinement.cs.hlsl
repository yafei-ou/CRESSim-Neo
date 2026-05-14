#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidVorticities);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidNeighborCounts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidNeighborPairs);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocitiesRW);

#define CRESSIM_FLUID_COMMON_HAS_MATERIAL_INDICES 1
#include "../../../include/physics/fluid/physics_fluid_common.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    if (CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        return;
    }

    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float confinement = max(fluidMaterial.vorticityConfinementDerived, 0.0);
    if (confinement <= kEpsilon)
    {
        return;
    }

    const float4 selfPositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float selfInvMass = selfPositionInvMass.w;
    if (selfInvMass <= kEpsilon)
    {
        return;
    }

    const float4 selfCurl4 = CRESSIM_SB_LOAD(g_FluidVorticities, particleIndex);
    const float3 selfCurl = selfCurl4.xyz;
    const float selfCurlMagnitude = selfCurl4.w;
    if (selfCurlMagnitude <= kEpsilon)
    {
        return;
    }

    const float3 selfPosition = selfPositionInvMass.xyz;
    const float smoothingRadius = max(fluidMaterial.smoothingRadius, 1.0e-4);
    float3 vorticityGrad = float3(0.0, 0.0, 0.0);
    const uint neighborCount = CRESSIM_SB_LOAD(g_FluidNeighborCounts, particleIndex);
    const uint neighborOffset = particleIndex * maxFluidNeighborhood;

    [loop]
    for (uint i = 0u; i < neighborCount; ++i)
    {
        const GpuParticleCandidatePair pair =
            CRESSIM_SB_LOAD(g_FluidNeighborPairs, neighborOffset + i);
        const uint neighborIndex = pair.indexB;
        if (!AreSameFluidMaterial(particleIndex, neighborIndex))
        {
            continue;
        }

        const float4 neighborPositionInvMass =
            CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, neighborIndex);
        const float3 delta = selfPosition - neighborPositionInvMass.xyz;
        const float distance = length(delta);
        if (distance <= kEpsilon || distance >= smoothingRadius)
        {
            continue;
        }

        const float neighborCurlMagnitude = CRESSIM_SB_LOAD(g_FluidVorticities, neighborIndex).w;
        vorticityGrad += neighborCurlMagnitude *
                         FluidSpikyKernelGradient(delta, distance, smoothingRadius);
    }

    const float gradLength = length(vorticityGrad);
    if (gradLength <= kEpsilon)
    {
        return;
    }

    const float3 vorticityForce = confinement * cross(vorticityGrad / gradLength, selfCurl);
    const float3 velocity = CRESSIM_SB_LOAD(g_ParticleVelocitiesRW, particleIndex).xyz;
    CRESSIM_SB_STORE(g_ParticleVelocitiesRW, particleIndex,
                     float4(velocity + dt * vorticityForce, 0.0));
}
