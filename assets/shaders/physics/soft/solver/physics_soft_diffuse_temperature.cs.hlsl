#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

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
           (flags & kSoftEdgeFracturedFlag) == 0u;
}

float sanitizeTemperature(float temperatureC, float bodyTemperatureC, float maximumTemperatureC)
{
    if (temperatureC != temperatureC)
    {
        return bodyTemperatureC;
    }
    return clamp(temperatureC, bodyTemperatureC, maximumTemperatureC);
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

    float neighbourTemperatureSum = 0.0;
    float neighbourWeightSum = 0.0;

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

        const GpuSoftParticleThermalState neighbourState =
            CRESSIM_SB_LOAD(g_ThermalStateRead, neighbourParticleIndex);
        const float neighbourTemperature = sanitizeTemperature(
            neighbourState.temperatureC, material.bodyTemperatureC, maxTemperatureC);
        const float restDistance = edge.referenceRestLength > kEpsilon
                                       ? edge.referenceRestLength
                                       : edge.restLength;
        const float distanceSquared = restDistance * restDistance;
        const float weight = 1.0 / max(distanceSquared, kEpsilon);
        neighbourTemperatureSum += neighbourTemperature * weight;
        neighbourWeightSum += weight;
    }

    const float averageNeighbourTemperature =
        neighbourWeightSum > kEpsilon
            ? neighbourTemperatureSum / max(neighbourWeightSum, kEpsilon)
            : currentTemperature;
    const float diffusionBlend =
        1.0 - exp(-max(material.diffusionRate, 0.0) * max(dt, 0.0));
    const float diffusedTemperature =
        lerp(currentTemperature, averageNeighbourTemperature, saturate(diffusionBlend));

    const float coolingBlend =
        1.0 - exp(-max(material.coolingRate, 0.0) * max(dt, 0.0));
    state.temperatureC =
        lerp(diffusedTemperature, material.bodyTemperatureC, saturate(coolingBlend));
    state.temperatureC = sanitizeTemperature(
        state.temperatureC, material.bodyTemperatureC, maxTemperatureC);
    state.damage = saturate(state.damage);
    state.waterFraction = saturate(state.waterFraction);
    state.charFraction = saturate(state.charFraction);

    CRESSIM_SB_STORE(g_ThermalStateWrite, particleIndex, state);
}
