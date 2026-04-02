#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuSoftRigidBroadPhaseParticle, g_SoftRigidBroadPhaseParticles);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedSoftRigidBroadPhaseKeys);

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_SoftParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleCollisionLayers);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftParticleCollisionMasks);

CRESSIM_STRUCTURED_BUFFER(float4, g_RigidSurfaceParticleWorldPositions);
CRESSIM_STRUCTURED_BUFFER(float, g_RigidSurfaceParticleSampleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleCollisionLayers);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidSurfaceParticleCollisionMasks);

CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftRigidCandidatePair, g_SoftRigidCandidatePairs);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_SoftRigidCandidatePairCount);

uint ComputeParticleGridCellKey(int gx, int gy, int gz)
{
    uint seed = uint(gx) * 73856093u;
    seed ^= uint(gy) * 19349663u;
    seed ^= uint(gz) * 83492791u;
    return seed;
}

uint FindFirstKeyIndex(uint targetKey, uint totalCount)
{
    uint left = 0u;
    uint right = totalCount;
    while (left < right)
    {
        const uint mid = left + ((right - left) >> 1u);
        const uint key = CRESSIM_SB_LOAD(g_SortedSoftRigidBroadPhaseKeys, mid).mortonCode;
        if (key < targetKey)
        {
            left = mid + 1u;
        }
        else
        {
            right = mid;
        }
    }
    return left;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint softIndex = dispatchThreadID.x;
    if (softIndex >= softParticleCount)
    {
        return;
    }

    const uint totalParticleCount = softParticleCount + rigidSurfaceParticleCount;
    const GpuSoftRigidBroadPhaseParticle selfEntry =
        CRESSIM_SB_LOAD(g_SoftRigidBroadPhaseParticles, softIndex);
    if (selfEntry.particleType != kSoftRigidBroadPhaseParticleTypeSoft)
    {
        return;
    }

    const float3 softPosition = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, softIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_SoftParticleRadii, softIndex);
    const uint softEnvironment = CRESSIM_SB_LOAD(g_SoftParticleEnvironmentIndices, softIndex);
    const uint softLayer = CRESSIM_SB_LOAD(g_SoftParticleCollisionLayers, softIndex);
    const uint softMask = CRESSIM_SB_LOAD(g_SoftParticleCollisionMasks, softIndex);

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
                uint sortedIndex = FindFirstKeyIndex(targetKey, totalParticleCount);

                while (sortedIndex < totalParticleCount)
                {
                    const GpuMortonCodeElement keyEntry =
                        CRESSIM_SB_LOAD(g_SortedSoftRigidBroadPhaseKeys, sortedIndex);
                    if (keyEntry.mortonCode != targetKey)
                    {
                        break;
                    }

                    const GpuSoftRigidBroadPhaseParticle candidateEntry =
                        CRESSIM_SB_LOAD(g_SoftRigidBroadPhaseParticles, keyEntry.elementIdx);
                    if (candidateEntry.cellX != cellX || candidateEntry.cellY != cellY ||
                        candidateEntry.cellZ != cellZ)
                    {
                        ++sortedIndex;
                        continue;
                    }

                    if (candidateEntry.particleType == kSoftRigidBroadPhaseParticleTypeSoft)
                    {
                        const uint otherSoftIndex = candidateEntry.particleIndex;
                        if (otherSoftIndex <= softIndex)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        const uint otherEnvironment =
                            CRESSIM_SB_LOAD(g_SoftParticleEnvironmentIndices, otherSoftIndex);
                        if (otherEnvironment != softEnvironment)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        const uint otherLayer =
                            CRESSIM_SB_LOAD(g_SoftParticleCollisionLayers, otherSoftIndex);
                        const uint otherMask =
                            CRESSIM_SB_LOAD(g_SoftParticleCollisionMasks, otherSoftIndex);
                        if ((softMask & otherLayer) == 0u || (otherMask & softLayer) == 0u)
                        {
                            ++sortedIndex;
                            continue;
                        }

                        const float3 otherPosition =
                            CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, otherSoftIndex).xyz;
                        const float combinedRadius =
                            softRadius + CRESSIM_SB_LOAD(g_SoftParticleRadii, otherSoftIndex);
                        const float3 deltaPos = otherPosition - softPosition;
                        if (dot(deltaPos, deltaPos) <= combinedRadius * combinedRadius)
                        {
                            uint pairIndex = 0u;
                            InterlockedAdd(g_SoftRigidCandidatePairCount[0], 1u, pairIndex);
                            if (pairIndex < softRigidCandidatePairCapacity)
                            {
                                GpuSoftRigidCandidatePair pair;
                                pair.pairType = kSoftRigidCandidatePairTypeSoftSoft;
                                pair.indexA = softIndex;
                                pair.indexB = otherSoftIndex;
                                pair.auxIndex = 0u;
                                CRESSIM_SB_STORE(g_SoftRigidCandidatePairs, pairIndex, pair);
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
                            InterlockedAdd(g_SoftRigidCandidatePairCount[0], 1u, pairIndex);
                            if (pairIndex < softRigidCandidatePairCapacity)
                            {
                                GpuSoftRigidCandidatePair pair;
                                pair.pairType = kSoftRigidCandidatePairTypeSoftRigid;
                                pair.indexA = softIndex;
                                pair.indexB = rigidBodyIndex;
                                pair.auxIndex = surfaceIndex;
                                CRESSIM_SB_STORE(g_SoftRigidCandidatePairs, pairIndex, pair);
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
