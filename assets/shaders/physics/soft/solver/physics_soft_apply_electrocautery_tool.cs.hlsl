#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwningSoftBodyIndices);
CRESSIM_STRUCTURED_BUFFER(GpuSoftThermalMaterial, g_SoftThermalMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuSoftParticleThermalState, g_ThermalStateRead);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftParticleThermalState, g_ThermalStateWrite);

float pointSegmentDistance(float3 samplePosition, float3 a, float3 b)
{
    const float3 ab = b - a;
    const float lenSq = dot(ab, ab);
    if (lenSq <= kEpsilon)
    {
        return length(samplePosition - a);
    }

    const float t = saturate(dot(samplePosition - a, ab) / lenSq);
    return length(samplePosition - (a + ab * t));
}

void activeTipSegment(float3 shaftA, float3 shaftB, float activeTipLength,
                      out float3 activeA, out float3 activeB)
{
    const float3 axis = shaftB - shaftA;
    const float lengthSq = dot(axis, axis);
    activeB = shaftB;
    if (lengthSq <= kEpsilon)
    {
        activeA = shaftB;
        return;
    }

    const float shaftLength = sqrt(lengthSq);
    activeA = shaftB - axis * saturate(activeTipLength / max(shaftLength, kEpsilon));
}

float distanceToElectrocauteryQuery(float3 samplePosition)
{
    float3 currentA;
    float3 currentB;
    float3 previousA;
    float3 previousB;
    activeTipSegment(electrocauteryToolTipA, electrocauteryToolTipB,
                     electrocauteryToolActiveTipLength, currentA, currentB);
    activeTipSegment(electrocauteryToolPreviousTipA, electrocauteryToolPreviousTipB,
                     electrocauteryToolActiveTipLength, previousA, previousB);

    float distanceToTool = pointSegmentDistance(samplePosition, currentA, currentB);
    distanceToTool = min(distanceToTool, pointSegmentDistance(samplePosition, previousA, previousB));
    distanceToTool = min(distanceToTool,
                         pointSegmentDistance(samplePosition, previousA, currentA));
    distanceToTool = min(distanceToTool,
                         pointSegmentDistance(samplePosition, previousB, currentB));
    return distanceToTool;
}

float sanitizeTemperature(float temperatureC, float bodyTemperatureC, float maximumTemperatureC)
{
    if (temperatureC != temperatureC)
    {
        return bodyTemperatureC;
    }
    return clamp(temperatureC, -273.15, maximumTemperatureC);
}

void electrocauteryModeThermalParams(out float powerDensity,
                                     out float heatingRadius,
                                     out float falloffExponent)
{
    powerDensity = 0.0;
    heatingRadius = 0.0;
    falloffExponent = 2.0;

    if (electrocauteryToolMode == kElectrocauteryToolModeCut)
    {
        powerDensity = electrocauteryCutPowerDensity;
        heatingRadius = electrocauteryCutHeatingRadius;
        falloffExponent = electrocauteryCutFalloffExponent;
    }
    else if (electrocauteryToolMode == kElectrocauteryToolModeCoagulation)
    {
        powerDensity = electrocauteryCoagulationPowerDensity;
        heatingRadius = electrocauteryCoagulationHeatingRadius;
        falloffExponent = electrocauteryCoagulationFalloffExponent;
    }
    else if (electrocauteryToolMode == kElectrocauteryToolModeBlend)
    {
        powerDensity = electrocauteryBlendPowerDensity;
        heatingRadius = electrocauteryBlendHeatingRadius;
        falloffExponent = electrocauteryBlendFalloffExponent;
    }

    heatingRadius = heatingRadius > 0.0 ? heatingRadius : electrocauteryToolHeatRadius;
    falloffExponent = max(falloffExponent, kEpsilon);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadId.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    GpuSoftParticleThermalState state = CRESSIM_SB_LOAD(g_ThermalStateRead, particleIndex);
    const uint softBodyIndex = CRESSIM_SB_LOAD(g_ParticleOwningSoftBodyIndices, particleIndex);
    if (softBodyIndex == 0xffffffffu)
    {
        CRESSIM_SB_STORE(g_ThermalStateWrite, particleIndex, state);
        return;
    }

    const GpuSoftThermalMaterial material =
        CRESSIM_SB_LOAD(g_SoftThermalMaterials, softBodyIndex);
    const float maxTemperatureC =
        max(material.maximumTemperatureC, material.bodyTemperatureC);
    state.temperatureC = sanitizeTemperature(
        state.temperatureC, material.bodyTemperatureC, maxTemperatureC);
    state.maximumTemperatureC = max(
        sanitizeTemperature(state.maximumTemperatureC, material.bodyTemperatureC, maxTemperatureC),
        state.temperatureC);
    state.arrheniusOmega = max(state.arrheniusOmega == state.arrheniusOmega
                                   ? state.arrheniusOmega
                                   : 0.0,
                               0.0);
    state.thermalDamage = saturate(state.thermalDamage);
    state.waterFraction = saturate(state.waterFraction);
    state.charLevel = saturate(state.charLevel);

    float modePowerDensity;
    float modeHeatingRadius;
    float modeFalloffExponent;
    electrocauteryModeThermalParams(modePowerDensity, modeHeatingRadius, modeFalloffExponent);

    const bool active = electrocauteryToolEnabled != 0u &&
                        electrocauteryToolMode != kElectrocauteryToolModeDisabled &&
                        (modePowerDensity > 0.0 ||
                         electrocauteryToolHeatingRateCPerSecond > 0.0) &&
                        modeHeatingRadius > 0.0;
    if (!active)
    {
        CRESSIM_SB_STORE(g_ThermalStateWrite, particleIndex, state);
        return;
    }

    const float3 particlePosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const float distanceToTool = distanceToElectrocauteryQuery(particlePosition);
    const float falloffWidth =
        max(modeHeatingRadius - electrocauteryToolAblationRadius, kEpsilon);
    float influence = 1.0 -
        saturate((distanceToTool - electrocauteryToolAblationRadius) / falloffWidth);
    influence = influence * influence * (3.0 - 2.0 * influence);
    influence = pow(saturate(influence), modeFalloffExponent);

    if (modePowerDensity > 0.0)
    {
        const float volumetricHeatCapacity =
            max(material.densityKgPerM3 * material.specificHeatJPerKgK, kEpsilon);
        state.temperatureC += dt * modePowerDensity * influence / volumetricHeatCapacity;
    }
    else
    {
        state.temperatureC += dt * electrocauteryToolHeatingRateCPerSecond * influence;
    }
    state.temperatureC = sanitizeTemperature(
        state.temperatureC, material.bodyTemperatureC, maxTemperatureC);
    state.maximumTemperatureC = max(state.maximumTemperatureC, state.temperatureC);
    CRESSIM_SB_STORE(g_ThermalStateWrite, particleIndex, state);
}
