#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/particle/physics_particle_grid.hlsli"
#include "physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleAdjacencyOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleAdjacencyCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleAdjacencyIndices);

CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_ParticleCandidatePairs);

#include "physics/particle/physics_particle_broad_phase_query.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint softIndex = dispatchThreadID.x;
    if (softIndex >= particleCount)
    {
        return;
    }

    const uint candidateCount = CRESSIM_SB_LOAD(g_CandidateCounts, softIndex);
    if (candidateCount == 0u)
    {
        return;
    }

    const GpuParticleBroadPhaseEntry selfEntry = CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, softIndex);
    const float3 softPosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, softIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_ParticleRadii, softIndex);
    const uint4 softMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, softIndex);
    const uint softEnvironment = softMetadata.x;
    const uint softPhase = softMetadata.y;
    const uint softLayer = softMetadata.z;
    const uint softMask = softMetadata.w;
    const uint softKind = CRESSIM_SB_LOAD(g_ParticleKinds, softIndex);
    uint writeIndex = CRESSIM_SB_LOAD(g_CandidateOffsets, softIndex);

    [loop]
    for (int dz = -1; dz <= 1; ++dz)
    {
        [loop]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [loop]
            for (int dx = -1; dx <= 1; ++dx)
            {
                const uint targetKey =
                    ComputeParticleGridCellKey(selfEntry.cellX + dx, selfEntry.cellY + dy,
                                               selfEntry.cellZ + dz);
                const GpuParticleCellRange range = FindCellRange(targetKey);
                if (range.cellKey == kInvalidIndex)
                {
                    continue;
                }

                for (uint sortedIndex = range.startIndex; sortedIndex < range.endIndex; ++sortedIndex)
                {
                    const GpuMortonCodeElement keyEntry =
                        CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, sortedIndex);
                    const GpuParticleBroadPhaseEntry candidateEntry =
                        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, keyEntry.elementIdx);
                    if (candidateEntry.cellX != selfEntry.cellX + dx ||
                        candidateEntry.cellY != selfEntry.cellY + dy ||
                        candidateEntry.cellZ != selfEntry.cellZ + dz)
                    {
                        continue;
                    }

                    uint otherSoftIndex = 0u;
                    if (!IsValidParticleParticleCandidate(
                            softIndex, softPosition, softRadius, softKind, softEnvironment,
                            softPhase, softLayer, softMask, candidateEntry, otherSoftIndex))
                    {
                        continue;
                    }

                    if (writeIndex >= particleCandidatePairCapacity)
                    {
                        return;
                    }

                    GpuParticleCandidatePair pair;
                    pair.pairType = kParticleCandidatePairTypeParticleParticle;
                    pair.indexA = softIndex;
                    pair.indexB = otherSoftIndex;
                    pair.auxIndex = 0u;
                    CRESSIM_SB_STORE(g_ParticleCandidatePairs, writeIndex, pair);
                    ++writeIndex;
                }
            }
        }
    }
}
