#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);

CRESSIM_RW_STRUCTURED_BUFFER(float, g_FluidDensities);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_FluidLambdas);

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
        CRESSIM_SB_STORE(g_FluidDensities, particleIndex, 0.0);
        CRESSIM_SB_STORE(g_FluidLambdas, particleIndex, 0.0);
        return;
    }

    const float4 selfPositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float selfInvMass = selfPositionInvMass.w;
    if (selfInvMass <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_FluidDensities, particleIndex, 0.0);
        CRESSIM_SB_STORE(g_FluidLambdas, particleIndex, 0.0);
        return;
    }

    const float selfMass = 1.0 / selfInvMass;
    const float3 selfPosition = selfPositionInvMass.xyz;
    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float restDensity = max(fluidMaterial.restDensity, 1.0);
    const float invRestDensity = 1.0 / restDensity;
    const float smoothingRadius = max(fluidMaterial.smoothingRadius, 1.0e-4);
    const uint4 selfMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, particleIndex);
    const uint selfEnvironment = selfMetadata.x;
    const uint selfLayer = selfMetadata.z;
    const uint selfMask = selfMetadata.w;
    const GpuParticleBroadPhaseEntry selfEntry =
        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, particleIndex);

    float density = selfMass * FluidCubicKernelZero(smoothingRadius);
    float3 gradI = float3(0.0, 0.0, 0.0);
    float sumGradSq = 0.0;

    [loop]
    for (int dz = -1; dz <= 1; ++dz)
    {
        [loop]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [loop]
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int targetCellX = selfEntry.cellX + dx;
                const int targetCellY = selfEntry.cellY + dy;
                const int targetCellZ = selfEntry.cellZ + dz;
                const uint targetKey =
                    ComputeParticleGridCellKey(targetCellX, targetCellY, targetCellZ);
                const GpuParticleCellRange range = FindParticleCellRange(targetKey);
                if (range.cellKey == kInvalidIndex)
                {
                    continue;
                }

                [loop]
                for (uint sortedIndex = range.startIndex; sortedIndex < range.endIndex; ++sortedIndex)
                {
                    const GpuMortonCodeElement keyEntry =
                        CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, sortedIndex);
                    const GpuParticleBroadPhaseEntry candidateEntry =
                        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, keyEntry.elementIdx);
                    if (candidateEntry.cellX != targetCellX || candidateEntry.cellY != targetCellY ||
                        candidateEntry.cellZ != targetCellZ)
                    {
                        continue;
                    }

                    const uint neighborIndex = candidateEntry.particleIndex;
                    if (!ShouldProcessFluidNeighbor(particleIndex, selfEnvironment, selfLayer,
                                                    selfMask, neighborIndex))
                    {
                        continue;
                    }

                    const float4 neighborPositionInvMass =
                        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, neighborIndex);
                    const float neighborInvMass = neighborPositionInvMass.w;
                    if (neighborInvMass <= kEpsilon)
                    {
                        continue;
                    }

                    const float3 delta = selfPosition - neighborPositionInvMass.xyz;
                    const float distance = length(delta);
                    if (distance >= smoothingRadius)
                    {
                        continue;
                    }

                    const float neighborMass = 1.0 / neighborInvMass;
                    density += neighborMass * FluidCubicKernel(distance, smoothingRadius);

                    const float3 gradC = (neighborMass * invRestDensity) *
                                         FluidCubicKernelGradient(delta, distance, smoothingRadius);
                    gradI += gradC;
                    sumGradSq += dot(gradC, gradC);
                }
            }
        }
    }

    sumGradSq += dot(gradI, gradI);
    const float constraint = max(density / restDensity - 1.0, 0.0);
    const float lambda =
        (constraint > 0.0) ? (-constraint / (sumGradSq + kFluidLambdaEpsilon)) : 0.0;

    CRESSIM_SB_STORE(g_FluidDensities, particleIndex, density);
    CRESSIM_SB_STORE(g_FluidLambdas, particleIndex, lambda);
}
