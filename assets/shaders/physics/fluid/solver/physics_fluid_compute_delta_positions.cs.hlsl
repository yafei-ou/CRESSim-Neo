#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(float, g_FluidConstraints);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidIterationDelta);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidSurfaceNormals);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidNeighborCounts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidNeighborPairs);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidDeltaPositions);

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
        CRESSIM_SB_STORE(g_FluidDeltaPositions, particleIndex, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float4 selfPositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float selfInvMass = selfPositionInvMass.w;
    if (selfInvMass <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_FluidDeltaPositions, particleIndex, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float3 selfAccumulatedDelta =
        (iterationIndex > 0u)
            ? CRESSIM_SB_LOAD(g_FluidIterationDelta, particleIndex).xyz
            : float3(0.0, 0.0, 0.0);
    const float3 selfPosition = selfPositionInvMass.xyz + selfAccumulatedDelta;
    const float selfConstraint = CRESSIM_SB_LOAD(g_FluidConstraints, particleIndex);
    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float smoothingRadius = max(fluidMaterial.smoothingRadius, 1.0e-4);
    const float cohesion = dt * max(fluidMaterial.cohesionDerived, 0.0);
    const float viscosity = dt * max(fluidMaterial.viscosityDerived, 0.0);
    const float surfaceTension = max(fluidMaterial.surfaceTensionDerived, 0.0);
    const float cflRadius = max(fluidMaterial.cflRadius, 0.0);
    float3 deltaPosition = float3(0.0, 0.0, 0.0);
    float interactionWeight = 0.0;
    const float3 selfSurfaceNormal =
        (surfaceTension > kEpsilon) ? CRESSIM_SB_LOAD(g_FluidSurfaceNormals, particleIndex).xyz
                                    : float3(0.0, 0.0, 0.0);
    const uint neighborCount = CRESSIM_SB_LOAD(g_FluidNeighborCounts, particleIndex);
    const uint neighborOffset = particleIndex * maxFluidNeighborhood;

    [loop]
    for (uint i = 0u; i < neighborCount; ++i)
    {
        const GpuParticleCandidatePair pair =
            CRESSIM_SB_LOAD(g_FluidNeighborPairs, neighborOffset + i);
        const uint neighborIndex = pair.indexB;
        const float4 neighborPositionInvMass =
            CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, neighborIndex);
        const float neighborInvMass = neighborPositionInvMass.w;
        if (neighborInvMass <= kEpsilon)
        {
            continue;
        }

        const float3 neighborAccumulatedDelta =
            (iterationIndex > 0u)
                ? CRESSIM_SB_LOAD(g_FluidIterationDelta, neighborIndex).xyz
                : float3(0.0, 0.0, 0.0);
        const float3 neighborPosition = neighborPositionInvMass.xyz + neighborAccumulatedDelta;

        const float3 delta = selfPosition - neighborPosition;
        const float distance = length(delta);
        if (distance >= smoothingRadius || distance <= kEpsilon)
        {
            continue;
        }

        const float neighborConstraint = CRESSIM_SB_LOAD(g_FluidConstraints, neighborIndex);
        const float3 nij = delta / distance;
        const float derivative = FluidSpikyKernelDerivative(distance, smoothingRadius);
        float3 pairDelta =
            -0.5 * (selfConstraint + neighborConstraint) * derivative * nij;

        if (AreSameFluidMaterial(particleIndex, neighborIndex))
        {
            const float3 relDelta = selfAccumulatedDelta - neighborAccumulatedDelta;

            if (cohesion > kEpsilon)
            {
                pairDelta -= nij * (cohesion *
                                    FluidCohesionKernel(distance, smoothingRadius,
                                                        fluidMaterial.cohesion1,
                                                        fluidMaterial.cohesion2));
            }

            if (viscosity > kEpsilon)
            {
                const float viscosityScale =
                    1.0 -
                    (1.0 / (1.0 + viscosity * FluidSpikyKernel(distance, smoothingRadius)));
                pairDelta -= viscosityScale * relDelta;
            }

            if (cflRadius > kEpsilon)
            {
                const float projected = dot(relDelta, nij);
                if (projected < -cflRadius)
                {
                    pairDelta -= nij * (projected + cflRadius) * 0.5;
                }
            }

            if (surfaceTension > kEpsilon)
            {
                const float3 neighborSurfaceNormal =
                    CRESSIM_SB_LOAD(g_FluidSurfaceNormals, neighborIndex).xyz;
                pairDelta -= dt * (selfSurfaceNormal - neighborSurfaceNormal);
            }
        }

        deltaPosition += pairDelta;
        interactionWeight += 1.0;
    }

    const float3 boundaryDelta =
        ComputeManualFluidBoundaryDelta(selfPosition, smoothingRadius, selfConstraint);

    const float3 packedDelta =
        deltaPosition * kFluidSolveCoefficient + boundaryDelta;

    CRESSIM_SB_STORE(g_FluidDeltaPositions, particleIndex,
                     float4(packedDelta, interactionWeight));
}
