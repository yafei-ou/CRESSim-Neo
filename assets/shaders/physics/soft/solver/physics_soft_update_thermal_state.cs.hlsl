#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwningSoftBodyIndices);
CRESSIM_STRUCTURED_BUFFER(GpuSoftThermalMaterial, g_SoftThermalMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuSoftParticleThermalState, g_ThermalStateRead);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftParticleThermalState, g_ThermalStateWrite);

float finiteOr(float value, float fallback)
{
    return value == value ? value : fallback;
}

float sanitizeTemperature(float temperatureC, float bodyTemperatureC, float maximumTemperatureC)
{
    return clamp(finiteOr(temperatureC, bodyTemperatureC), -273.15, maximumTemperatureC);
}

float smoothActivation(float startValue, float fullValue, float value)
{
    return fullValue > startValue + kEpsilon
               ? smoothstep(startValue, fullValue, value)
               : (value >= startValue ? 1.0 : 0.0);
}

float accumulateThermalOmega(GpuSoftThermalMaterial material,
                             float existingOmega,
                             float temperatureC)
{
    const float previousOmega = max(finiteOr(existingOmega, 0.0), 0.0);
    if (material.damageModel == kThermalDamageModelArrhenius)
    {
        const float gasConstantJPerMolK = 8.314462618;
        const float temperatureK = max(temperatureC + 273.15, 1.0);
        const float logRate =
            material.logArrheniusA -
            material.activationEnergyJPerMol / (gasConstantJPerMolK * temperatureK);
        const float rate = exp(clamp(logRate, -80.0, 80.0));
        return min(previousOmega + max(dt, 0.0) * rate, 1.0e20);
    }

    const float normalizedThermalActivation =
        smoothActivation(material.damageStartTemperatureC,
                         material.damageFullTemperatureC,
                         temperatureC);
    const float omegaRate =
        max(material.damageRate, 0.0) * normalizedThermalActivation;
    return min(previousOmega + omegaRate * max(dt, 0.0), 1.0e20);
}

float normalizedThermalDamage(float omega)
{
    return saturate(1.0 - exp(-max(finiteOr(omega, 0.0), 0.0)));
}

float reduceWaterFraction(GpuSoftThermalMaterial material,
                          float existingWaterFraction,
                          float temperatureC,
                          float waterLossScale)
{
    const float transitionWidth = max(material.evaporationTransitionWidthC, 0.0);
    const float evaporationActivation =
        smoothActivation(material.evaporationStartTemperatureC,
                         material.evaporationStartTemperatureC + transitionWidth,
                         temperatureC);
    const float evaporationRate =
        max(material.evaporationRate, 0.0) * max(waterLossScale, 0.0) *
        evaporationActivation;
    const float previousWaterFraction = saturate(finiteOr(existingWaterFraction, 1.0));
    const float updatedWaterFraction =
        previousWaterFraction * exp(-evaporationRate * max(dt, 0.0));
    return saturate(min(previousWaterFraction, updatedWaterFraction));
}

float accumulateCharFraction(GpuSoftThermalMaterial material,
                             float existingCharFraction,
                             float waterFraction,
                             float temperatureC,
                             float charScale)
{
    const float charTemperatureActivation =
        smoothActivation(material.charStartTemperatureC,
                         material.charFullTemperatureC,
                         temperatureC);
    const float dryness = 1.0 - saturate(waterFraction);
    const float charRate =
        max(material.charRate, 0.0) * max(charScale, 0.0) *
        charTemperatureActivation * dryness;
    const float previousCharFraction = saturate(finiteOr(existingCharFraction, 0.0));
    const float updatedCharFraction =
        1.0 - (1.0 - previousCharFraction) * exp(-charRate * max(dt, 0.0));
    return saturate(max(previousCharFraction, updatedCharFraction));
}

float activeElectrocauteryCharScale()
{
    if (electrocauteryToolEnabled == 0u)
    {
        return 1.0;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeCut)
    {
        return electrocauteryCutCharScale;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeCoagulation)
    {
        return electrocauteryCoagulationCharScale;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeBlend)
    {
        return electrocauteryBlendCharScale;
    }
    return 1.0;
}

float activeElectrocauteryWaterLossScale()
{
    if (electrocauteryToolEnabled == 0u)
    {
        return 1.0;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeCut)
    {
        return electrocauteryCutWaterLossScale;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeCoagulation)
    {
        return electrocauteryCoagulationWaterLossScale;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeBlend)
    {
        return electrocauteryBlendWaterLossScale;
    }
    return 1.0;
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
    state.arrheniusOmega =
        accumulateThermalOmega(material, state.arrheniusOmega, state.temperatureC);
    state.thermalDamage = normalizedThermalDamage(state.arrheniusOmega);
    state.waterFraction = reduceWaterFraction(
        material, state.waterFraction, state.temperatureC, activeElectrocauteryWaterLossScale());
    state.charLevel = accumulateCharFraction(
        material, state.charLevel, state.waterFraction, state.maximumTemperatureC,
        activeElectrocauteryCharScale());

    state.temperatureC = sanitizeTemperature(
        state.temperatureC, material.bodyTemperatureC, maxTemperatureC);
    state.maximumTemperatureC = max(
        sanitizeTemperature(state.maximumTemperatureC, material.bodyTemperatureC, maxTemperatureC),
        state.temperatureC);
    state.arrheniusOmega = max(finiteOr(state.arrheniusOmega, 0.0), 0.0);
    state.thermalDamage = saturate(state.thermalDamage);
    state.waterFraction = saturate(state.waterFraction);
    state.charLevel = saturate(state.charLevel);

    CRESSIM_SB_STORE(g_ThermalStateWrite, particleIndex, state);
}
