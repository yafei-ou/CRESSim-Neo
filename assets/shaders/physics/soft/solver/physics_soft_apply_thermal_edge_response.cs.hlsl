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

float segmentSegmentDistance(float3 p1, float3 q1, float3 p2, float3 q2)
{
    const float3 d1 = q1 - p1;
    const float3 d2 = q2 - p2;
    const float3 r = p1 - p2;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);

    float s = 0.0;
    float t = 0.0;
    if (a <= kEpsilon && e <= kEpsilon)
    {
        return length(p1 - p2);
    }
    if (a <= kEpsilon)
    {
        t = saturate(f / max(e, kEpsilon));
    }
    else
    {
        const float c = dot(d1, r);
        if (e <= kEpsilon)
        {
            s = saturate(-c / max(a, kEpsilon));
        }
        else
        {
            const float b = dot(d1, d2);
            const float denom = a * e - b * b;
            if (abs(denom) > kEpsilon)
            {
                s = saturate((b * f - c * e) / denom);
            }
            t = (b * s + f) / e;
            if (t < 0.0)
            {
                t = 0.0;
                s = saturate(-c / a);
            }
            else if (t > 1.0)
            {
                t = 1.0;
                s = saturate((b - c) / a);
            }
        }
    }

    const float3 c1 = p1 + d1 * s;
    const float3 c2 = p2 + d2 * t;
    return length(c1 - c2);
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

bool edgeIntersectsElectrocauteryCutSweep(float3 positionA, float3 positionB)
{
    if (electrocauteryToolEnabled == 0u ||
        electrocauteryToolMode != kElectrocauteryToolModeCut ||
        electrocauteryCutThermalCutEnabled == 0u)
    {
        return false;
    }

    float3 currentA;
    float3 currentB;
    float3 previousA;
    float3 previousB;
    activeTipSegment(electrocauteryToolTipA, electrocauteryToolTipB,
                     electrocauteryToolActiveTipLength, currentA, currentB);
    activeTipSegment(electrocauteryToolPreviousTipA, electrocauteryToolPreviousTipB,
                     electrocauteryToolActiveTipLength, previousA, previousB);

    const float cutRadius =
        electrocauteryToolAblationRadius > kEpsilon
            ? electrocauteryToolAblationRadius
            : electrocauteryCutHeatingRadius;
    const float sweepRadius = max(cutRadius, kEpsilon);
    float edgeDistance = segmentSegmentDistance(positionA, positionB, currentA, currentB);
    edgeDistance = min(edgeDistance, segmentSegmentDistance(positionA, positionB,
                                                           previousA, previousB));
    edgeDistance = min(edgeDistance, segmentSegmentDistance(positionA, positionB,
                                                           previousA, currentA));
    edgeDistance = min(edgeDistance, segmentSegmentDistance(positionA, positionB,
                                                           previousB, currentB));
    return edgeDistance <= sweepRadius;
}

float activeElectrocauteryShrinkageScale()
{
    if (electrocauteryToolEnabled == 0u)
    {
        return 1.0;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeCut)
    {
        return electrocauteryCutShrinkageScale;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeCoagulation)
    {
        return electrocauteryCoagulationShrinkageScale;
    }
    if (electrocauteryToolMode == kElectrocauteryToolModeBlend)
    {
        return electrocauteryBlendShrinkageScale;
    }
    return 1.0;
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
    const float edgeArrheniusOmega =
        max(max(thermalA.arrheniusOmega, 0.0), max(thermalB.arrheniusOmega, 0.0));
    const float edgeThermalDamage =
        max(saturate(thermalA.thermalDamage), saturate(thermalB.thermalDamage));
    const float edgeWaterFraction =
        min(saturate(thermalA.waterFraction), saturate(thermalB.waterFraction));
    const float shrinkActivation =
        smoothActivation(material.shrinkDamageStart,
                         material.shrinkDamageFull,
                         edgeThermalDamage);
    const float shrinkageScale = max(activeElectrocauteryShrinkageScale(), 0.0);

    const float referenceRestLength =
        edge.referenceRestLength > kEpsilon ? edge.referenceRestLength : edge.restLength;
    const float targetRestLength =
        referenceRestLength *
        (1.0 - saturate(material.maximumShrinkage * shrinkageScale) * shrinkActivation);
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
    const bool sufficientlyDamagedByOmega =
        edgeArrheniusOmega >= max(material.thermalCutOmega, 0.0);
    const bool sufficientlyDamagedByNormalizedDamage =
        edgeThermalDamage >= saturate(material.thermalCutDamageThreshold);
    const bool sufficientlyDry =
        edgeWaterFraction <= saturate(material.thermalCutWaterThreshold);
    const bool intersectsCutSweep =
        edgeIntersectsElectrocauteryCutSweep(positionA, positionB);

    if (intersectsCutSweep &&
        (sufficientlyDamagedByOmega || sufficientlyDamagedByNormalizedDamage ||
         sufficientlyDry))
    {
        edge.flags |= kSoftEdgeThermallySeverableFlag;
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
    }

    CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
}
