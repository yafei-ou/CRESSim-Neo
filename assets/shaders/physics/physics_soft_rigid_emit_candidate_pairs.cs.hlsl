#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_SoftParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SoftParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleAdjacencyOffsets);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleAdjacencyCounts);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleAdjacencyIndices);

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidSurfaceParticleWorldPositions);
CRESSIM_STRUCTURED_BUFFER(float, g_RigidSurfaceParticleSampleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleCollisionLayers);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleCollisionMasks);

CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftCandidatePair, g_SoftCandidatePairs);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_SoftCandidatePairCount);

uint ComputeParticleGridCellKey(int gx, int gy, int gz)
{
    uint seed = uint(gx) * 73856093u;
    seed ^= uint(gy) * 19349663u;
    seed ^= uint(gz) * 83492791u;
    return seed;
}

GpuParticleCellRange FindCellRange(uint targetKey)
{
    GpuParticleCellRange missingRange;
    missingRange.cellKey = kInvalidIndex;
    missingRange.startIndex = 0u;
    missingRange.endIndex = 0u;
    missingRange.reserved0 = 0u;

    if (softCellRangeCapacity == 0u)
    {
        return missingRange;
    }

    uint lo = 0u;
    uint hi = softCellRangeCapacity;
    [loop]
    while (lo < hi)
    {
        const uint mid = lo + (hi - lo) / 2u;
        const GpuParticleCellRange range = CRESSIM_SB_LOAD(g_ParticleCellRanges, mid);
        if (range.cellKey < targetKey)
        {
            lo = mid + 1u;
        }
        else
        {
            hi = mid;
        }
    }

    if (lo < softCellRangeCapacity)
    {
        const GpuParticleCellRange range = CRESSIM_SB_LOAD(g_ParticleCellRanges, lo);
        if (range.cellKey == targetKey)
        {
            return range;
        }
    }

    return missingRange;
}

bool IsAdjacentSoftParticle(uint particleIndex, uint candidateIndex)
{
    const uint neighborOffset = CRESSIM_SB_LOAD(g_SoftParticleAdjacencyOffsets, particleIndex);
    const uint neighborCount = CRESSIM_SB_LOAD(g_SoftParticleAdjacencyCounts, particleIndex);
    [loop]
    for (uint i = 0u; i < neighborCount; ++i)
    {
        if (CRESSIM_SB_LOAD(g_SoftParticleAdjacencyIndices, neighborOffset + i) == candidateIndex)
        {
            return true;
        }
    }
    return false;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint softIndex = dispatchThreadID.x;
    if (softIndex >= softParticleCount)
    {
        return;
    }

    const GpuParticleBroadPhaseEntry selfEntry =
        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, softIndex);
    if (selfEntry.particleType != kParticleBroadPhaseEntryTypeSoft)
    {
        return;
    }

    const float3 softPosition = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, softIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_SoftParticleRadii, softIndex);
    const uint4 softMetadata = CRESSIM_SB_LOAD(g_SoftParticleBroadPhaseMetadata, softIndex);
    const uint softEnvironment = softMetadata.x;
    const uint softPhase = softMetadata.y;
    const uint softLayer = softMetadata.z;
    const uint softMask = softMetadata.w;

    uint seenRigidBodies[16];
    uint seenRigidCount = 0u;

    [loop]
    for (int dz = -1; dz <= 1; ++dz)
    {
        [loop]
        for (int dy = -1; dy <= 1; ++dy)
        {
            [loop]
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int cellX = selfEntry.cellX + dx;
                const int cellY = selfEntry.cellY + dy;
                const int cellZ = selfEntry.cellZ + dz;
                const uint targetKey = ComputeParticleGridCellKey(cellX, cellY, cellZ);
                const GpuParticleCellRange range = FindCellRange(targetKey);
                if (range.cellKey == kInvalidIndex)
                {
                    continue;
                }

                uint sortedIndex = range.startIndex;
                while (sortedIndex < range.endIndex)
                {
                    const GpuMortonCodeElement keyEntry =
                        CRESSIM_SB_LOAD(g_SortedParticleBroadPhaseKeys, sortedIndex);
                    const GpuParticleBroadPhaseEntry candidateEntry =
                        CRESSIM_SB_LOAD(g_ParticleBroadPhaseEntries, keyEntry.elementIdx);
                    if (candidateEntry.cellX != cellX || candidateEntry.cellY != cellY ||
                        candidateEntry.cellZ != cellZ)
                    {
                        ++sortedIndex;
                        continue;
                    }

                    if (candidateEntry.particleType == kParticleBroadPhaseEntryTypeSoft)
                    {
                        const uint otherSoftIndex = candidateEntry.particleIndex;
                        if (otherSoftIndex <= softIndex)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        const uint4 otherMetadata =
                            CRESSIM_SB_LOAD(g_SoftParticleBroadPhaseMetadata, otherSoftIndex);
                        const uint otherEnvironment = otherMetadata.x;
                        if (otherEnvironment != softEnvironment)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        const uint otherPhase = otherMetadata.y;
                        const uint otherLayer = otherMetadata.z;
                        const uint otherMask = otherMetadata.w;
                        if ((softMask & otherLayer) == 0u || (otherMask & softLayer) == 0u)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        if (SoftParticlePhaseGroup(softPhase) == SoftParticlePhaseGroup(otherPhase))
                        {
                            const bool selfCollideA = SoftParticlePhaseSelfCollideEnabled(softPhase);
                            const bool selfCollideB =
                                SoftParticlePhaseSelfCollideEnabled(otherPhase);
                            if (!selfCollideA || !selfCollideB)
                            {
                                ++sortedIndex;
                                continue;
                            }

                            if (IsAdjacentSoftParticle(softIndex, otherSoftIndex))
                            {
                                ++sortedIndex;
                                continue;
                            }
                        }

                        const float3 otherPosition =
                            CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, otherSoftIndex).xyz;
                        const float combinedRadius =
                            softRadius + CRESSIM_SB_LOAD(g_SoftParticleRadii, otherSoftIndex);
                        const float3 deltaPos = otherPosition - softPosition;
                        if (dot(deltaPos, deltaPos) <= combinedRadius * combinedRadius)
                        {
                            uint pairIndex = 0u;
                            InterlockedAdd(g_SoftCandidatePairCount[0], 1u, pairIndex);
                            if (pairIndex < softCandidatePairCapacity)
                            {
                                GpuSoftCandidatePair pair;
                                pair.pairType = kSoftCandidatePairTypeSoftSoft;
                                pair.indexA = softIndex;
                                pair.indexB = otherSoftIndex;
                                pair.auxIndex = 0u;
                                CRESSIM_SB_STORE(g_SoftCandidatePairs, pairIndex, pair);
                            }
                        }
                    }
                    else
                    {
                        const uint surfaceIndex = candidateEntry.particleIndex;
                        const uint surfaceEnvironment =
                            CRESSIM_SB_LOAD(g_RigidSurfaceParticleEnvironmentIndices, surfaceIndex);
                        if (surfaceEnvironment != softEnvironment)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        const uint surfaceLayer =
                            CRESSIM_SB_LOAD(g_RigidSurfaceParticleCollisionLayers, surfaceIndex);
                        const uint surfaceMask =
                            CRESSIM_SB_LOAD(g_RigidSurfaceParticleCollisionMasks, surfaceIndex);
                        if ((softMask & surfaceLayer) == 0u || (surfaceMask & softLayer) == 0u)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        const float3 surfacePosition =
                            CRESSIM_SB_LOAD(g_RigidSurfaceParticleWorldPositions, surfaceIndex).xyz;
                        const float combinedRadius =
                            softRadius +
                            CRESSIM_SB_LOAD(g_RigidSurfaceParticleSampleRadii, surfaceIndex);
                        const float3 deltaPos = surfacePosition - softPosition;
                        if (dot(deltaPos, deltaPos) <= combinedRadius * combinedRadius)
                        {
                            const uint rigidBodyIndex = candidateEntry.ownerIndex;
                            bool alreadySeen = false;
                            [unroll]
                            for (uint i = 0u; i < 16u; ++i)
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
                            if (!alreadySeen)
                            {
                                if (seenRigidCount < 16u)
                                {
                                    seenRigidBodies[seenRigidCount] = rigidBodyIndex;
                                    ++seenRigidCount;
                                }

                                uint pairIndex = 0u;
                            InterlockedAdd(g_SoftCandidatePairCount[0], 1u, pairIndex);
                            if (pairIndex < softCandidatePairCapacity)
                            {
                                GpuSoftCandidatePair pair;
                                pair.pairType = kSoftCandidatePairTypeSoftRigid;
                                pair.indexA = softIndex;
                                pair.indexB = rigidBodyIndex;
                                pair.auxIndex = surfaceIndex;
                                CRESSIM_SB_STORE(g_SoftCandidatePairs, pairIndex, pair);
                            }
                            }
                        }
                    }

                    ++sortedIndex;
                }
            }
        }
    }
}
