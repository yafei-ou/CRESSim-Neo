#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
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
           (flags & kSoftEdgeThermalCutFlag) == 0u &&
           (flags & kSoftEdgeFracturedFlag) == 0u;
}

float smoothActivation(float startValue, float fullValue, float value)
{
    return fullValue > startValue + kEpsilon
               ? smoothstep(startValue, fullValue, value)
               : (value >= startValue ? 1.0 : 0.0);
}

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

float ablationInfluence(float3 samplePosition)
{
    if (electrocauteryToolEnabled == 0u ||
        electrocauteryToolMode != kElectrocauteryToolModeCut ||
        electrocauteryToolAblationRadius <= kEpsilon)
    {
        return 0.0;
    }

    float influence =
        1.0 - saturate(distanceToElectrocauteryQuery(samplePosition) /
                       electrocauteryToolAblationRadius);
    return influence * influence * (3.0 - 2.0 * influence);
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
    const float edgeWaterFraction =
        min(saturate(thermalA.waterFraction), saturate(thermalB.waterFraction));
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
        lerp(1.0, max(material.maximumComplianceMultiplier, 1.0), edgeThermalDamage);
    edge.compliance =
        max(edge.referenceCompliance, 0.0) * complianceMultiplier;

    const float3 positionA =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleA).xyz;
    const float3 positionB =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleB).xyz;
    const float3 midpoint = 0.5 * (positionA + positionB);
    const bool sufficientlyDamaged =
        edgeThermalDamage >= saturate(material.thermalCutDamageThreshold);
    const bool sufficientlyDry =
        edgeWaterFraction <= saturate(material.thermalCutWaterThreshold);
    const bool closeToActiveTool =
        ablationInfluence(midpoint) >= saturate(electrocauteryToolAblationInfluenceThreshold);

    if (sufficientlyDamaged && sufficientlyDry && closeToActiveTool)
    {
        edge.flags =
            (edge.flags | kSoftEdgeCutFlag | kSoftEdgeThermalCutFlag |
             kSoftEdgeDisabledFlag) &
            ~kSoftEdgeActiveFlag;
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
    }

    CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
}
