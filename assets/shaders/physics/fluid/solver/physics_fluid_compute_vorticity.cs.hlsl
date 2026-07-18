#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleVelocitiesRW);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidNeighborCounts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidNeighborPairs);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidVorticities);

#define CRESSIM_FLUID_COMMON_HAS_MATERIAL_INDICES 1
#include "physics_fluid_common.hlsli"

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
        CRESSIM_SB_STORE(g_FluidVorticities, particleIndex, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    if (max(fluidMaterial.vorticityConfinementDerived, 0.0) <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_FluidVorticities, particleIndex, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float4 selfPositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float selfInvMass = selfPositionInvMass.w;
    if (selfInvMass <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_FluidVorticities, particleIndex, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float3 selfPosition = selfPositionInvMass.xyz;
    const float3 selfVelocity = CRESSIM_SB_LOAD(g_ParticleVelocitiesRW, particleIndex).xyz;
    const float smoothingRadius = max(fluidMaterial.smoothingRadius, 1.0e-4);
    float3 curl = float3(0.0, 0.0, 0.0);
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

        const float3 neighborVelocity =
            CRESSIM_SB_LOAD(g_ParticleVelocitiesRW, neighborIndex).xyz;
        curl += cross(neighborVelocity - selfVelocity,
                      FluidSpikyKernelGradient(delta, distance, smoothingRadius));
    }

    CRESSIM_SB_STORE(g_FluidVorticities, particleIndex, float4(curl, length(curl)));
}
