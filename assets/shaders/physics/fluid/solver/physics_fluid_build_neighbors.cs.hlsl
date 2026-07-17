#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_grid.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_FluidMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(GpuFluidMaterial, g_FluidMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidIterationDelta);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_CandidateCounts);

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidNeighborPairs);

#include "physics/fluid/physics_fluid_neighbor_build.hlsli"

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

    const float4 selfPositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    if (selfPositionInvMass.w <= kEpsilon)
    {
        return;
    }

    if (maxFluidNeighborhood == 0u)
    {
        CRESSIM_SB_STORE(g_CandidateCounts, particleIndex, 0u);
        return;
    }

    const uint candidateOffset = particleIndex * maxFluidNeighborhood;

    const float3 selfAccumulatedDelta =
        (iterationIndex > 0u)
            ? CRESSIM_SB_LOAD(g_FluidIterationDelta, particleIndex).xyz
            : float3(0.0, 0.0, 0.0);
    const float3 selfPosition = selfPositionInvMass.xyz + selfAccumulatedDelta;
    const uint fluidMaterialIndex = CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex);
    const GpuFluidMaterial fluidMaterial = CRESSIM_SB_LOAD(g_FluidMaterials, fluidMaterialIndex);
    const float smoothingRadius = max(fluidMaterial.smoothingRadius, 1.0e-4);
    const uint4 selfMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, particleIndex);
    const uint selfEnvironment = selfMetadata.x;
    const uint selfLayer = selfMetadata.z;
    const uint selfMask = selfMetadata.w;
    const GpuParticleBroadPhaseEntry selfEntry =
        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, particleIndex);

    uint writeIndex = candidateOffset;
    uint candidateCount = 0u;

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
                    if (neighborPositionInvMass.w <= kEpsilon)
                    {
                        continue;
                    }

                    const float3 neighborAccumulatedDelta =
                        (iterationIndex > 0u)
                            ? CRESSIM_SB_LOAD(g_FluidIterationDelta, neighborIndex).xyz
                            : float3(0.0, 0.0, 0.0);
                    const float3 delta =
                        selfPosition - (neighborPositionInvMass.xyz + neighborAccumulatedDelta);
                    if (length(delta) >= smoothingRadius)
                    {
                        continue;
                    }

                    if (candidateCount >= maxFluidNeighborhood)
                    {
                        continue;
                    }

                    GpuParticleCandidatePair pair;
                    pair.pairType = kParticleCandidatePairTypeParticleParticle;
                    pair.indexA = particleIndex;
                    pair.indexB = neighborIndex;
                    pair.auxIndex = 0u;
                    CRESSIM_SB_STORE(g_FluidNeighborPairs, writeIndex, pair);
                    ++writeIndex;
                    ++candidateCount;
                }
            }
        }
    }

    // This path currently clamps each particle to a fixed neighborhood budget and
    // can silently truncate dense fluid neighborhoods. Keep that behavior for now
    // until we add a dedicated overflow meta/reporting path.
    CRESSIM_SB_STORE(g_CandidateCounts, particleIndex, candidateCount);
}
