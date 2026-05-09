#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/fluid/physics_fluid_common.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuParticleBroadPhaseEntry, g_ParticleBroadPhaseEntries);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCellRange, g_ParticleCellRanges);
CRESSIM_STRUCTURED_BUFFER(GpuMortonCodeElement, g_SortedParticleBroadPhaseKeys);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(float, g_FluidRestDensities);
CRESSIM_STRUCTURED_BUFFER(float, g_FluidSmoothingRadii);

CRESSIM_RW_STRUCTURED_BUFFER(float, g_FluidDensities);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_FluidLambdas);

GpuParticleCellRange FindParticleCellRange(uint targetKey)
{
    GpuParticleCellRange missingRange;
    missingRange.cellKey = kInvalidIndex;
    missingRange.startIndex = 0u;
    missingRange.endIndex = 0u;
    missingRange.reserved0 = 0u;

    if (particleCellRangeCapacity == 0u)
    {
        return missingRange;
    }

    uint lo = 0u;
    uint hi = particleCellRangeCapacity;
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

    if (lo < particleCellRangeCapacity)
    {
        const GpuParticleCellRange range = CRESSIM_SB_LOAD(g_ParticleCellRanges, lo);
        if (range.cellKey == targetKey)
        {
            return range;
        }
    }

    return missingRange;
}

bool ShouldProcessFluidNeighbor(uint selfIndex, uint selfEnvironment, uint selfLayer, uint selfMask,
                                uint otherIndex)
{
    if (otherIndex == selfIndex)
    {
        return false;
    }

    if (CRESSIM_SB_LOAD(g_ParticleKinds, otherIndex) != kParticleKindFluid)
    {
        return false;
    }

    const uint4 otherMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, otherIndex);
    if (otherMetadata.x != selfEnvironment)
    {
        return false;
    }

    const uint otherLayer = otherMetadata.z;
    const uint otherMask = otherMetadata.w;
    return (selfMask & otherLayer) != 0u && (otherMask & selfLayer) != 0u;
}

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
    const float restDensity = max(CRESSIM_SB_LOAD(g_FluidRestDensities, particleIndex), 1.0);
    const float smoothingRadius = max(CRESSIM_SB_LOAD(g_FluidSmoothingRadii, particleIndex), 1.0e-4);
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
                const uint targetKey =
                    ComputeParticleGridCellKey(selfEntry.cellX + dx, selfEntry.cellY + dy,
                                               selfEntry.cellZ + dz);
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
                    if (candidateEntry.cellX != selfEntry.cellX + dx ||
                        candidateEntry.cellY != selfEntry.cellY + dy ||
                        candidateEntry.cellZ != selfEntry.cellZ + dz)
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

                    const float3 gradC =
                        (neighborMass / restDensity) *
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
        (constraint > 0.0)
            ? (-constraint / max(sumGradSq + kFluidLambdaEpsilon, kFluidLambdaEpsilon))
            : 0.0;

    CRESSIM_SB_STORE(g_FluidDensities, particleIndex, density);
    CRESSIM_SB_STORE(g_FluidLambdas, particleIndex, lambda);
}
