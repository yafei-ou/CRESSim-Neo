#ifndef CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI

#include "../physics_particle_dispatch_constants.hlsli"
#include "../core/physics_base.hlsli"
#include "../particle/physics_particle_types.hlsli"

static const float kFluidConstraintEpsilon = 1.0e-6;
static const float kFluidSolveCoefficient = 0.01;

bool AreSameFluidMaterial(uint particleIndex, uint otherIndex)
{
    return CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex) ==
           CRESSIM_SB_LOAD(g_FluidMaterialIndices, otherIndex);
}

float FluidSpikyKernel(float distance, float smoothingRadius)
{
    if (distance >= smoothingRadius)
    {
        return 0.0;
    }

    const float oneMinusQ = 1.0 - distance / smoothingRadius;
    const float h3 = smoothingRadius * smoothingRadius * smoothingRadius;
    const float k = 15.0 / (3.14159265359 * h3);

    return k * oneMinusQ * oneMinusQ;
}

float FluidSpikyKernelDerivative(float distance, float smoothingRadius)
{
    if (distance <= kEpsilon || distance >= smoothingRadius)
    {
        return 0.0;
    }

    const float oneMinusQ = 1.0 - distance / smoothingRadius;
    const float h3 = smoothingRadius * smoothingRadius * smoothingRadius;
    const float k = 15.0 / (3.14159265359 * h3);
    return -2.0 * k * oneMinusQ / smoothingRadius;
}

float3 FluidSpikyKernelGradient(float3 delta, float distance, float smoothingRadius)
{
    if (distance <= kEpsilon || distance >= smoothingRadius)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float dWdr = FluidSpikyKernelDerivative(distance, smoothingRadius);
    return dWdr * (delta / distance);
}

float FluidCohesionKernel(float distance, float smoothingRadius, float cohesion1,
                          float cohesion2)
{
    if (distance <= kEpsilon || distance >= smoothingRadius)
    {
        return 0.0;
    }

    const float q = distance / smoothingRadius;
    return cohesion1 * q * q * q + cohesion2 * q * q - 1.0;
}

#endif // CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI
