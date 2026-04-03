#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_SoftParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SoftParticleBroadPhaseMetadata);

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidSurfaceParticleWorldPositions);
CRESSIM_STRUCTURED_BUFFER(float, g_RigidSurfaceParticleSampleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleCollisionLayers);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleCollisionMasks);

CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_CandidateOffsets);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftCandidatePair, g_SoftCandidatePairs);

#include "physics/include/physics_soft_neighbor_base.hlsli"
#include "physics/include/physics_soft_rigid_neighbor.hlsli"

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
    const uint softLayer = softMetadata.z;
    const uint softMask = softMetadata.w;
    uint seenRigidBodies[kSoftRigidDedupCacheSize];
    uint seenRigidCount = 0u;
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

                    uint rigidBodyIndex = 0u;
                    uint surfaceIndex = 0u;
                    if (!IsValidSoftRigidCandidate(softPosition, softRadius, softEnvironment,
                                                   softLayer, softMask, candidateEntry,
                                                   rigidBodyIndex, surfaceIndex))
                    {
                        continue;
                    }

                    bool alreadySeen = false;
                    [unroll]
                    for (uint i = 0u; i < kSoftRigidDedupCacheSize; ++i)
                    {
                        if (i >= seenRigidCount)
                        {
                            break;
                        }
                        if (seenRigidBodies[i] == rigidBodyIndex)
                        {
                            alreadySeen = true;
                            break;
                        }
                    }
                    if (alreadySeen)
                    {
                        continue;
                    }

                    if (seenRigidCount < kSoftRigidDedupCacheSize)
                    {
                        seenRigidBodies[seenRigidCount] = rigidBodyIndex;
                        ++seenRigidCount;
                    }

                    if (writeIndex >= softCandidatePairCapacity)
                    {
                        return;
                    }

                    GpuSoftCandidatePair pair;
                    pair.pairType = kSoftCandidatePairTypeSoftRigid;
                    pair.indexA = softIndex;
                    pair.indexB = rigidBodyIndex;
                    pair.auxIndex = surfaceIndex;
                    CRESSIM_SB_STORE(g_SoftCandidatePairs, writeIndex, pair);
                    ++writeIndex;
                }
            }
        }
    }
}
