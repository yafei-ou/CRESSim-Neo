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
    return clamp(finiteOr(temperatureC, bodyTemperatureC), bodyTemperatureC, maximumTemperatureC);
}

float smoothActivation(float startValue, float fullValue, float value)
{
    return fullValue > startValue + kEpsilon
               ? smoothstep(startValue, fullValue, value)
               : (value >= startValue ? 1.0 : 0.0);
}

float accumulateThresholdRateDamage(GpuSoftThermalMaterial material,
                                    float existingDamage,
                                    float temperatureC)
{
    const float normalizedThermalActivation =
        smoothActivation(material.damageStartTemperatureC,
                         material.damageFullTemperatureC,
                         temperatureC);
    const float damageRate =
        max(material.damageRate, 0.0) * normalizedThermalActivation;
    const float previousDamage = saturate(finiteOr(existingDamage, 0.0));
    const float updatedDamage =
        1.0 - (1.0 - previousDamage) * exp(-damageRate * max(dt, 0.0));
    return saturate(max(previousDamage, updatedDamage));
}

float accumulateThermalDamage(GpuSoftThermalMaterial material,
                              float existingDamage,
                              float temperatureC)
{
    if (material.damageModel == kThermalDamageModelArrhenius)
    {
        return accumulateThresholdRateDamage(material, existingDamage, temperatureC);
    }

    return accumulateThresholdRateDamage(material, existingDamage, temperatureC);
}

float reduceWaterFraction(GpuSoftThermalMaterial material,
                          float existingWaterFraction,
                          float temperatureC)
{
    const float transitionWidth = max(material.evaporationTransitionWidthC, 0.0);
    const float evaporationActivation =
        smoothActivation(material.evaporationStartTemperatureC,
                         material.evaporationStartTemperatureC + transitionWidth,
                         temperatureC);
    const float evaporationRate =
        max(material.evaporationRate, 0.0) * evaporationActivation;
    const float previousWaterFraction = saturate(finiteOr(existingWaterFraction, 1.0));
    const float updatedWaterFraction =
        previousWaterFraction * exp(-evaporationRate * max(dt, 0.0));
    return saturate(min(previousWaterFraction, updatedWaterFraction));
}

float accumulateCharFraction(GpuSoftThermalMaterial material,
                             float existingCharFraction,
                             float waterFraction,
                             float temperatureC)
{
    const float charTemperatureActivation =
        smoothActivation(material.charStartTemperatureC,
                         material.charFullTemperatureC,
                         temperatureC);
    const float dryness = 1.0 - saturate(waterFraction);
    const float charRate =
        max(material.charRate, 0.0) * charTemperatureActivation * dryness;
    const float previousCharFraction = saturate(finiteOr(existingCharFraction, 0.0));
    const float updatedCharFraction =
        1.0 - (1.0 - previousCharFraction) * exp(-charRate * max(dt, 0.0));
    return saturate(max(previousCharFraction, updatedCharFraction));
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
    state.damage = accumulateThermalDamage(material, state.damage, state.temperatureC);
    state.waterFraction = reduceWaterFraction(
        material, state.waterFraction, state.temperatureC);
    state.charFraction = accumulateCharFraction(
        material, state.charFraction, state.waterFraction, state.temperatureC);

    state.temperatureC = sanitizeTemperature(
        state.temperatureC, material.bodyTemperatureC, maxTemperatureC);
    state.damage = saturate(state.damage);
    state.waterFraction = saturate(state.waterFraction);
    state.charFraction = saturate(state.charFraction);

    CRESSIM_SB_STORE(g_ThermalStateWrite, particleIndex, state);
}
