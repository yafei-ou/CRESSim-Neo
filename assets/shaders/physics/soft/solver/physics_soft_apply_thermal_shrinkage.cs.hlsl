#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwningSoftBodyIndices);
CRESSIM_STRUCTURED_BUFFER(GpuSoftThermalMaterial, g_SoftThermalMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuSoftParticleThermalState, g_ThermalStateRead);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);

bool isEdgeActive(uint flags)
{
    return (flags & kSoftEdgeActiveFlag) != 0u &&
           (flags & kSoftEdgeDisabledFlag) == 0u &&
           (flags & kSoftEdgeCutFlag) == 0u &&
           (flags & kSoftEdgeFracturedFlag) == 0u;
}

float smoothActivation(float startValue, float fullValue, float value)
{
    return fullValue > startValue + kEpsilon
               ? smoothstep(startValue, fullValue, value)
               : (value >= startValue ? 1.0 : 0.0);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint edgeIndex = dispatchThreadID.x;
    if (edgeIndex >= softEdgeCount)
    {
        return;
    }

    GpuSoftEdge edge = CRESSIM_SB_LOAD(g_SoftEdges, edgeIndex);
    if (!isEdgeActive(edge.flags))
    {
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
        return;
    }

    const uint softBodyIndex = CRESSIM_SB_LOAD(g_ParticleOwningSoftBodyIndices, edge.particleA);
    if (softBodyIndex == 0xffffffffu)
    {
        CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
        return;
    }

    const GpuSoftThermalMaterial material =
        CRESSIM_SB_LOAD(g_SoftThermalMaterials, softBodyIndex);
    const GpuSoftParticleThermalState thermalA =
        CRESSIM_SB_LOAD(g_ThermalStateRead, edge.particleA);
    const GpuSoftParticleThermalState thermalB =
        CRESSIM_SB_LOAD(g_ThermalStateRead, edge.particleB);
    const float edgeThermalDamage =
        max(saturate(thermalA.damage), saturate(thermalB.damage));
    const float averageDamage =
        0.5 * (saturate(thermalA.damage) + saturate(thermalB.damage));
    const float shrinkActivation =
        smoothActivation(material.shrinkDamageStart,
                         material.shrinkDamageFull,
                         averageDamage);

    const float referenceRestLength =
        edge.referenceRestLength > kEpsilon ? edge.referenceRestLength : edge.restLength;
    const float targetRestLength =
        referenceRestLength * (1.0 - saturate(material.maximumShrinkage) * shrinkActivation);
    const float shrinkBlend =
        1.0 - exp(-max(material.shrinkageRate, 0.0) * max(dt, 0.0));
    edge.restLength =
        lerp(edge.restLength, max(targetRestLength, kEpsilon), saturate(shrinkBlend));

    const float failureScale =
        lerp(1.0, saturate(material.minimumFailureThresholdScale), edgeThermalDamage);
    edge.failureThreshold =
        edge.referenceFailureThreshold * failureScale;

    const float cutResistanceScale =
        lerp(1.0, saturate(material.minimumCutResistanceScale), edgeThermalDamage);
    edge.cutResistance =
        max(edge.referenceCutResistance * cutResistanceScale, kEpsilon);

    const float complianceMultiplier =
        lerp(1.0, max(material.maximumComplianceMultiplier, 1.0), averageDamage);
    edge.compliance =
        max(edge.referenceCompliance, 0.0) * complianceMultiplier;

    CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
}
