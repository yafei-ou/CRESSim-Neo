#include "include/physics/physics_soft_dispatch_constants.hlsli"
#include "include/physics/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_SoftParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SoftParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleAdjacencyOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleAdjacencyCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleAdjacencyIndices);

CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftCandidatePair, g_SoftCandidatePairs);

#include "include/physics/physics_soft_neighbor_base.hlsli"
#include "include/physics/physics_soft_soft_neighbor.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint softIndex = dispatchThreadID.x;
    if (softIndex >= softParticleCount)
    {
        return;
    }

    const uint candidateCount = CRESSIM_SB_LOAD(g_CandidateCounts, softIndex);
    if (candidateCount == 0u)
    {
        return;
    }

    const GpuParticleBroadPhaseEntry selfEntry = CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, softIndex);
    const float3 softPosition = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, softIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_SoftParticleRadii, softIndex);
    const uint4 softMetadata = CRESSIM_SB_LOAD(g_SoftParticleBroadPhaseMetadata, softIndex);
    const uint softEnvironment = softMetadata.x;
    const uint softPhase = softMetadata.y;
    const uint softLayer = softMetadata.z;
    const uint softMask = softMetadata.w;
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
                    if (!IsValidSoftSoftCandidate(softIndex, softPosition, softRadius, softEnvironment,
                                                  softPhase, softLayer, softMask, candidateEntry,
                                                  otherSoftIndex))
                    {
                        continue;
                    }

                    if (writeIndex >= softCandidatePairCapacity)
                    {
                        return;
                    }

                    GpuSoftCandidatePair pair;
                    pair.pairType = kSoftCandidatePairTypeSoftSoft;
                    pair.indexA = softIndex;
                    pair.indexB = otherSoftIndex;
                    pair.auxIndex = 0u;
                    CRESSIM_SB_STORE(g_SoftCandidatePairs, writeIndex, pair);
                    ++writeIndex;
                }
            }
        }
    }
}
