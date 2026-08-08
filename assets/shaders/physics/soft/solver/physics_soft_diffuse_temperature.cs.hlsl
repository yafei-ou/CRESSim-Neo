#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwningSoftBodyIndices);
CRESSIM_STRUCTURED_BUFFER(GpuSoftThermalMaterial, g_SoftThermalMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleEdgeRanges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftIncidentEdge, g_ParticleIncidentEdges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftParticleThermalState, g_ThermalStateRead);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftParticleThermalState, g_ThermalStateWrite);

bool isEdgeActive(uint flags)
{
    return (flags & kSoftEdgeActiveFlag) != 0u &&
           (flags & kSoftEdgeDisabledFlag) == 0u &&
           (flags & kSoftEdgeCutFlag) == 0u &&
           (flags & kSoftEdgeThermalCutFlag) == 0u &&
           (flags & kSoftEdgeFracturedFlag) == 0u;
}

float sanitizeTemperature(float temperatureC, float bodyTemperatureC, float maximumTemperatureC)
{
    if (temperatureC != temperatureC)
    {
        return bodyTemperatureC;
    }
    return clamp(temperatureC, -273.15, maximumTemperatureC);
}

float harmonicMeanNonNegative(float a, float b)
{
    a = max(a, 0.0);
    b = max(b, 0.0);
    return (a + b) > kEpsilon ? (2.0 * a * b) / (a + b) : 0.0;
}

float particleMassFromInverseMass(float inverseMass)
{
    return inverseMass > kEpsilon ? 1.0 / inverseMass : 0.0;
}

float particleVolumeFromMass(float particleMass, float densityKgPerM3)
{
    return particleMass > kEpsilon
               ? particleMass / max(densityKgPerM3, kEpsilon)
               : 0.0;
}

float effectiveInterfaceVolume(float volumeA, float volumeB, float fallbackVolume)
{
    if (volumeA > kEpsilon && volumeB > kEpsilon)
    {
        return harmonicMeanNonNegative(volumeA, volumeB);
    }
    if (volumeA > kEpsilon)
    {
        return volumeA;
    }
    if (volumeB > kEpsilon)
    {
        return volumeB;
    }
    return max(fallbackVolume, 0.0);
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
    const float currentTemperature = sanitizeTemperature(
        state.temperatureC, material.bodyTemperatureC, maxTemperatureC);
    const float4 particlePositionInvMass =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float particleMassFromState =
        particleMassFromInverseMass(particlePositionInvMass.w);
    const float particleVolumeFromState =
        particleVolumeFromMass(particleMassFromState, material.densityKgPerM3);

    float conductionPower = 0.0;
    float conductionConductance = 0.0;
    float interfaceVolumeSum = 0.0;
    uint activeThermalEdgeCount = 0u;

    const GpuSoftConstraintRange edgeRange =
        CRESSIM_SB_LOAD(g_ParticleEdgeRanges, particleIndex);
    for (uint adjacencyIndex = edgeRange.start;
         adjacencyIndex < edgeRange.start + edgeRange.count;
         ++adjacencyIndex)
    {
        const GpuSoftIncidentEdge incidentEdge =
            CRESSIM_SB_LOAD(g_ParticleIncidentEdges, adjacencyIndex);
        const GpuSoftEdge edge = CRESSIM_SB_LOAD(g_SoftEdges, incidentEdge.edgeIndex);
        if (!isEdgeActive(edge.flags))
        {
            continue;
        }

        const uint neighbourParticleIndex = GetSoftEdgeOtherParticle(edge, particleIndex);
        if (neighbourParticleIndex >= particleCount || neighbourParticleIndex == particleIndex)
        {
            continue;
        }

        const uint neighbourSoftBodyIndex =
            CRESSIM_SB_LOAD(g_ParticleOwningSoftBodyIndices, neighbourParticleIndex);
        if (neighbourSoftBodyIndex == 0xffffffffu)
        {
            continue;
        }

        const GpuSoftThermalMaterial neighbourMaterial =
            CRESSIM_SB_LOAD(g_SoftThermalMaterials, neighbourSoftBodyIndex);
        const GpuSoftParticleThermalState neighbourState =
            CRESSIM_SB_LOAD(g_ThermalStateRead, neighbourParticleIndex);
        const float4 neighbourPositionInvMass =
            CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, neighbourParticleIndex);
        const float neighbourMaxTemperatureC =
            max(neighbourMaterial.maximumTemperatureC, neighbourMaterial.bodyTemperatureC);
        const float neighbourTemperature = sanitizeTemperature(
            neighbourState.temperatureC,
            neighbourMaterial.bodyTemperatureC,
            neighbourMaxTemperatureC);
        const float restDistance = edge.referenceRestLength > kEpsilon
                                       ? edge.referenceRestLength
                                       : edge.restLength;
        const float lengthScaleMetersPerWorldUnit =
            max(harmonicMeanNonNegative(material.metersPerWorldUnit,
                                        neighbourMaterial.metersPerWorldUnit),
                kEpsilon);
        const float distanceMeters =
            max(restDistance * lengthScaleMetersPerWorldUnit, kEpsilon);
        const float symmetricConductivity =
            harmonicMeanNonNegative(material.thermalConductivityWPerMK,
                                    neighbourMaterial.thermalConductivityWPerMK);
        const float neighbourMass =
            particleMassFromInverseMass(neighbourPositionInvMass.w);
        const float neighbourVolume =
            particleVolumeFromMass(neighbourMass, neighbourMaterial.densityKgPerM3);
        const float fallbackVolume = distanceMeters * distanceMeters * distanceMeters;
        const float interfaceVolume =
            effectiveInterfaceVolume(particleVolumeFromState, neighbourVolume, fallbackVolume);
        const float interfaceArea = pow(max(interfaceVolume, kEpsilon), 2.0 / 3.0);
        const float conductance =
            max(symmetricConductivity, 0.0) * interfaceArea / distanceMeters;
        conductionPower += conductance * (neighbourTemperature - currentTemperature);
        conductionConductance += conductance;
        interfaceVolumeSum += interfaceVolume;
        ++activeThermalEdgeCount;
    }

    const float averageInterfaceVolume =
        activeThermalEdgeCount != 0u
            ? interfaceVolumeSum / float(activeThermalEdgeCount)
            : 0.0;
    const float particleVolume =
        particleVolumeFromState > kEpsilon
            ? particleVolumeFromState
            : max(averageInterfaceVolume, 1.0e-12);
    const float particleMass =
        particleMassFromState > kEpsilon
            ? particleMassFromState
            : max(material.densityKgPerM3 * particleVolume, kEpsilon);

    const float perfusionConductance =
        max(material.bloodDensityKgPerM3, 0.0) *
        max(material.bloodSpecificHeatJPerKgK, 0.0) *
        max(material.bloodPerfusionPerSecond, 0.0) *
        particleVolume;
    const float perfusion =
        perfusionConductance * (material.bloodTemperatureC - currentTemperature);
    const float metabolicPower = material.metabolicHeatWPerM3 * particleVolume;
    const float particleHeatCapacity =
        max(particleMass * material.specificHeatJPerKgK, kEpsilon);
    const float thermalDt = max(dt, 0.0);
    const float stableConductanceLimit =
        thermalDt > kEpsilon ? 0.45 * particleHeatCapacity / thermalDt : 0.0;
    const float dynamicConductance = conductionConductance + perfusionConductance;
    const float stabilityScale =
        dynamicConductance > stableConductanceLimit && dynamicConductance > kEpsilon
            ? stableConductanceLimit / dynamicConductance
            : 1.0;
    const float temperatureRate =
        ((conductionPower + perfusion) * saturate(stabilityScale) + metabolicPower) /
        particleHeatCapacity;
    state.temperatureC = currentTemperature + thermalDt * temperatureRate;
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

    CRESSIM_SB_STORE(g_ThermalStateWrite, particleIndex, state);
}
