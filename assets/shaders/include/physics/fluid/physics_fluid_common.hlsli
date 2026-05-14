#ifndef CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI

#include "../physics_particle_dispatch_constants.hlsli"
#include "../core/physics_base.hlsli"
#include "../particle/physics_particle_types.hlsli"

static const float kFluidConstraintEpsilon = 1.0e-6;
static const float kFluidSolveCoefficient = 0.01;

// Current virtual-box boundary model used by the fluid solver. This is an
// open-top axis-aligned container in world space that contributes support in
// the density pass and matching correction in the delta pass, with clamp-based
// cleanup left to the apply pass.
static const bool kManualFluidBoundaryEnabled = true;
static const float3 kManualFluidBoundaryMin = float3(-2.2, -0.85, -2.2);
static const float3 kManualFluidBoundaryMax = float3(2.2, 8.0, 2.2);
static const float kManualFluidBoundaryDensityScale = 0.12;
static const float kManualFluidBoundaryDeltaScale = 0.08;

#ifdef CRESSIM_FLUID_COMMON_HAS_MATERIAL_INDICES
bool AreSameFluidMaterial(uint particleIndex, uint otherIndex)
{
    return CRESSIM_SB_LOAD(g_FluidMaterialIndices, particleIndex) ==
           CRESSIM_SB_LOAD(g_FluidMaterialIndices, otherIndex);
}
#endif

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

void AccumulateBoundaryDensityContribution(float3 ghostDelta, float smoothingRadius,
                                           float surfaceTension, inout float density,
                                           inout float3 surfaceNormal)
{
    const float distance = length(ghostDelta);
    if (distance >= smoothingRadius)
    {
        return;
    }

    density += FluidSpikyKernel(distance, smoothingRadius) * kManualFluidBoundaryDensityScale;

    // Keep boundary support out of the free-surface normal estimate. Injecting
    // wall/floor normals here creates artificial curvature near the container.
}

float3 ComputeBoundaryDeltaContribution(float3 ghostDelta, float smoothingRadius,
                                        float selfConstraint)
{
    const float distance = length(ghostDelta);
    if (distance <= kEpsilon || distance >= smoothingRadius)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float3 nij = ghostDelta / distance;
    const float derivative = FluidSpikyKernelDerivative(distance, smoothingRadius);
    return -selfConstraint * derivative * nij;
}

void AccumulateManualBoundaryFaceDensity(float faceDistance, float3 ghostDelta,
                                         float smoothingRadius, float surfaceTension,
                                         inout float density, inout float3 surfaceNormal)
{
    if (faceDistance < 0.0 || faceDistance >= smoothingRadius)
    {
        return;
    }

    AccumulateBoundaryDensityContribution(ghostDelta, smoothingRadius, surfaceTension, density,
                                          surfaceNormal);
}

void AccumulateManualBoundaryFaceDelta(float faceDistance, float3 ghostDelta,
                                       float smoothingRadius, float selfConstraint,
                                       inout float3 boundaryDelta)
{
    if (faceDistance < 0.0 || faceDistance >= smoothingRadius)
    {
        return;
    }

    boundaryDelta += ComputeBoundaryDeltaContribution(ghostDelta, smoothingRadius,
                                                      max(selfConstraint, 0.0));
}

void AccumulateManualFluidBoundaryDensity(float3 selfPosition, float smoothingRadius,
                                          float surfaceTension, inout float density,
                                          inout float3 surfaceNormal)
{
    if (!kManualFluidBoundaryEnabled)
    {
        return;
    }

    const float leftDistance = selfPosition.x - kManualFluidBoundaryMin.x;
    const float rightDistance = kManualFluidBoundaryMax.x - selfPosition.x;
    const float floorDistance = selfPosition.y - kManualFluidBoundaryMin.y;
    const float backDistance = selfPosition.z - kManualFluidBoundaryMin.z;
    const float frontDistance = kManualFluidBoundaryMax.z - selfPosition.z;

    AccumulateManualBoundaryFaceDensity(leftDistance, float3(2.0 * leftDistance, 0.0, 0.0),
                                        smoothingRadius, surfaceTension, density,
                                        surfaceNormal);
    AccumulateManualBoundaryFaceDensity(rightDistance, float3(-2.0 * rightDistance, 0.0, 0.0),
                                        smoothingRadius, surfaceTension, density,
                                        surfaceNormal);
    AccumulateManualBoundaryFaceDensity(floorDistance, float3(0.0, 2.0 * floorDistance, 0.0),
                                        smoothingRadius, surfaceTension, density,
                                        surfaceNormal);
    AccumulateManualBoundaryFaceDensity(backDistance, float3(0.0, 0.0, 2.0 * backDistance),
                                        smoothingRadius, surfaceTension, density,
                                        surfaceNormal);
    AccumulateManualBoundaryFaceDensity(frontDistance, float3(0.0, 0.0, -2.0 * frontDistance),
                                        smoothingRadius, surfaceTension, density,
                                        surfaceNormal);
}

float3 ComputeManualFluidBoundaryDelta(float3 selfPosition, float smoothingRadius,
                                       float selfConstraint)
{
    if (!kManualFluidBoundaryEnabled)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float leftDistance = selfPosition.x - kManualFluidBoundaryMin.x;
    const float rightDistance = kManualFluidBoundaryMax.x - selfPosition.x;
    const float floorDistance = selfPosition.y - kManualFluidBoundaryMin.y;
    const float backDistance = selfPosition.z - kManualFluidBoundaryMin.z;
    const float frontDistance = kManualFluidBoundaryMax.z - selfPosition.z;

    float3 boundaryDelta = float3(0.0, 0.0, 0.0);
    AccumulateManualBoundaryFaceDelta(leftDistance, float3(2.0 * leftDistance, 0.0, 0.0),
                                      smoothingRadius, selfConstraint, boundaryDelta);
    AccumulateManualBoundaryFaceDelta(rightDistance, float3(-2.0 * rightDistance, 0.0, 0.0),
                                      smoothingRadius, selfConstraint, boundaryDelta);
    AccumulateManualBoundaryFaceDelta(floorDistance, float3(0.0, 2.0 * floorDistance, 0.0),
                                      smoothingRadius, selfConstraint, boundaryDelta);
    AccumulateManualBoundaryFaceDelta(backDistance, float3(0.0, 0.0, 2.0 * backDistance),
                                      smoothingRadius, selfConstraint, boundaryDelta);
    AccumulateManualBoundaryFaceDelta(frontDistance, float3(0.0, 0.0, -2.0 * frontDistance),
                                      smoothingRadius, selfConstraint, boundaryDelta);
    return boundaryDelta * kManualFluidBoundaryDeltaScale;
}

float3 ClampManualFluidBoundaryPosition(float3 position, float particleRadius)
{
    if (!kManualFluidBoundaryEnabled)
    {
        return position;
    }

    const float3 paddedMin = kManualFluidBoundaryMin + float3(particleRadius, particleRadius,
                                                              particleRadius);
    const float3 paddedMax = float3(kManualFluidBoundaryMax.x - particleRadius,
                                    kManualFluidBoundaryMax.y,
                                    kManualFluidBoundaryMax.z - particleRadius);
    float3 clamped = position;
    clamped.x = clamp(clamped.x, paddedMin.x, paddedMax.x);
    clamped.y = max(clamped.y, paddedMin.y);
    clamped.z = clamp(clamped.z, paddedMin.z, paddedMax.z);
    return clamped;
}

#endif // CRESSIM_NEO_PHYSICS_FLUID_COMMON_HLSLI
