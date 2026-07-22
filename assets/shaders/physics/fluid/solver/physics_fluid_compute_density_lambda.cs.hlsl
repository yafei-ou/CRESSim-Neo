#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_rigid_broad_phase_types.hlsli"
#include "physics_rigid_types.hlsli"
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidIterationDelta);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidNeighborCounts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidNeighborPairs);
CRESSIM_STRUCTURED_BUFFER(uint2, g_FluidBoundaryCandidateRanges);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidBoundaryCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);
CRESSIM_STRUCTURED_BUFFER(GpuColliderGeometryData, g_ColliderGeometryData);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidSurfaceNormalConstraints);

#define CRESSIM_FLUID_COMMON_HAS_MATERIAL_INDICES 1
#include "physics_fluid_common.hlsli"
#include "physics_fluid_boundary_common.hlsli"

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
        CRESSIM_SB_STORE(g_FluidSurfaceNormalConstraints, particleIndex,
                         float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float4 selfPositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float selfInvMass = selfPositionInvMass.w;
    if (selfInvMass <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_FluidSurfaceNormalConstraints, particleIndex,
                         float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float3 selfAccumulatedDelta =
        (iterationIndex > 0u)
            ? CRESSIM_SB_LOAD(g_FluidIterationDelta, particleIndex).xyz
            : float3(0.0, 0.0, 0.0);
    const float3 selfPosition = selfPositionInvMass.xyz + selfAccumulatedDelta;
    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float restDensity = max(fluidMaterial.restDensity, 1.0);
    // The CPU derives this scale from the smoothing radius. It becomes much smaller
    // at smaller particle scales, so an absolute shader floor breaks scale similarity.
    const float densityConstraintScale = fluidMaterial.densityConstraintScaleDerived;
    const float smoothingRadius = max(fluidMaterial.smoothingRadius, 1.0e-4);
    const float surfaceTension = max(fluidMaterial.surfaceTensionDerived, 0.0);
    float density = 0.0;
    float3 surfaceNormal = float3(0.0, 0.0, 0.0);
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
        const float3 neighborAccumulatedDelta =
            (iterationIndex > 0u)
                ? CRESSIM_SB_LOAD(g_FluidIterationDelta, neighborIndex).xyz
                : float3(0.0, 0.0, 0.0);
        const float3 neighborPosition = neighborPositionInvMass.xyz + neighborAccumulatedDelta;
        const float3 delta = selfPosition - neighborPosition;
        const float distance = length(delta);
        if (distance >= smoothingRadius)
        {
            continue;
        }

        density += FluidSpikyKernel(distance, smoothingRadius);

        if (surfaceTension > kEpsilon)
        {
            surfaceNormal += FluidSpikyKernelGradient(delta, distance, smoothingRadius);
        }
    }

    AccumulateFluidBoundaryDensityFromCandidates(particleIndex, selfPosition, smoothingRadius,
                                                 surfaceTension, density, surfaceNormal);

    const float constraint =
        max(density - restDensity, -restDensity * max(kFluidMaxUnderDensityRatio, 0.0)) *
        densityConstraintScale;

    if (surfaceTension > kEpsilon)
    {
        surfaceNormal *= surfaceTension;
        CRESSIM_SB_STORE(g_FluidSurfaceNormalConstraints, particleIndex,
                         float4(surfaceNormal, constraint));
    }
    else
    {
        CRESSIM_SB_STORE(g_FluidSurfaceNormalConstraints, particleIndex,
                         float4(0.0, 0.0, 0.0, constraint));
    }
}
