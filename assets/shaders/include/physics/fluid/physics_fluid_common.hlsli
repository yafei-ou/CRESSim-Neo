#ifndef CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI

#include "../physics_particle_dispatch_constants.hlsli"
#include "../core/physics_base.hlsli"
#include "../particle/physics_particle_grid.hlsli"
#include "../particle/physics_particle_types.hlsli"

static const float kFluidLambdaEpsilon = 1.0e-6;

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

float FluidCubicKernel(float distance, float smoothingRadius)
{
    if (distance >= smoothingRadius)
    {
        return 0.0;
    }

    const float q = distance / smoothingRadius;
    const float h3 = smoothingRadius * smoothingRadius * smoothingRadius;
    const float k = 8.0 / (3.14159265359 * h3);

    if (q <= 0.5)
    {
        const float q2 = q * q;
        const float q3 = q2 * q;
        return k * (6.0 * q3 - 6.0 * q2 + 1.0);
    }

    const float oneMinusQ = 1.0 - q;
    return k * (2.0 * oneMinusQ * oneMinusQ * oneMinusQ);
}

float FluidCubicKernelZero(float smoothingRadius)
{
    return FluidCubicKernel(0.0, smoothingRadius);
}

float3 FluidCubicKernelGradient(float3 delta, float distance, float smoothingRadius)
{
    if (distance <= kEpsilon || distance >= smoothingRadius)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float q = distance / smoothingRadius;
    const float h3 = smoothingRadius * smoothingRadius * smoothingRadius;
    const float l = 48.0 / (3.14159265359 * h3);
    const float3 gradQ = delta / (distance * smoothingRadius);

    if (q <= 0.5)
    {
        return l * q * (3.0 * q - 2.0) * gradQ;
    }

    const float oneMinusQ = 1.0 - q;
    return l * (-(oneMinusQ * oneMinusQ)) * gradQ;
}

#endif // CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI
