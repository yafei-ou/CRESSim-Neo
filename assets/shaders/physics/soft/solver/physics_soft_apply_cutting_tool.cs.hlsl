#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_SoftEdgeToolCounters);

static const uint kToolCandidateCounter = 0u;
static const uint kNewlyCutCounter = 1u;
static const uint kAlreadyDisabledCounter = 2u;
static const uint kActiveAfterCutCounter = 3u;

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

bool segmentIntersectsAabb(float3 p0, float3 p1, float3 halfExtents)
{
    const float3 direction = p1 - p0;
    float tMin = 0.0;
    float tMax = 1.0;

    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (abs(direction[axis]) <= kEpsilon)
        {
            if (p0[axis] < -halfExtents[axis] || p0[axis] > halfExtents[axis])
            {
                return false;
            }
        }
        else
        {
            const float inverseDirection = 1.0 / direction[axis];
            float t0 = (-halfExtents[axis] - p0[axis]) * inverseDirection;
            float t1 = (halfExtents[axis] - p0[axis]) * inverseDirection;
            if (t0 > t1)
            {
                const float temporary = t0;
                t0 = t1;
                t1 = temporary;
            }

            tMin = max(tMin, t0);
            tMax = min(tMax, t1);
            if (tMin > tMax)
            {
                return false;
            }
        }
    }

    return true;
}

float3 toBladeLocal(float3 worldPosition)
{
    const float3 offset = worldPosition - cuttingToolBladeCenter;
    return float3(dot(offset, cuttingToolBladeAxisU),
                  dot(offset, cuttingToolBladeAxisV),
                  dot(offset, cuttingToolBladeNormal));
}

bool edgeIntersectsCuttingTool(float3 positionA, float3 positionB)
{
    if (cuttingToolShape == kCuttingToolShapeBlade)
    {
        if (cuttingToolBladeHalfLength <= 0.0 || cuttingToolBladeHalfDepth <= 0.0 ||
            cuttingToolBladeHalfThickness <= 0.0)
        {
            return false;
        }

        const float3 localA = toBladeLocal(positionA);
        const float3 localB = toBladeLocal(positionB);
        const float3 bladeHalfExtents = float3(cuttingToolBladeHalfLength,
                                               cuttingToolBladeHalfDepth,
                                               cuttingToolBladeHalfThickness);
        return segmentIntersectsAabb(localA, localB, bladeHalfExtents);
    }

    if (cuttingToolRadius <= 0.0)
    {
        return false;
    }

    const float toolDistance =
        segmentSegmentDistance(positionA, positionB, cuttingToolTipA, cuttingToolTipB);
    return toolDistance <= cuttingToolRadius;
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
    const bool sampleDiagnostics = reserved1 > 0u && reserved0 + 1u == reserved1;
    uint ignoredCounterValue = 0u;

    if ((edge.flags & kSoftEdgeActiveFlag) == 0u ||
        (edge.flags & kSoftEdgeDisabledFlag) != 0u)
    {
        if (sampleDiagnostics)
        {
            InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kAlreadyDisabledCounter),
                           1u, ignoredCounterValue);
        }
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
        return;
    }

    const bool toolEnabled = cuttingToolEnabled != 0u &&
                             (cuttingToolStrength > 0.0 || cuttingToolInstantCut != 0u);
    if (toolEnabled)
    {
        const float3 positionA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleA).xyz;
        const float3 positionB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleB).xyz;
        if (edgeIntersectsCuttingTool(positionA, positionB))
        {
            InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kToolCandidateCounter),
                           1u, ignoredCounterValue);
            edge.damage = saturate(edge.damage + cuttingToolStrength * dt);
            const float cutResistance =
                max(edge.cutResistance * cuttingToolCutResistanceScale, kEpsilon);
            if (cuttingToolInstantCut != 0u || edge.damage >= cutResistance)
            {
                InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kNewlyCutCounter),
                               1u, ignoredCounterValue);
                edge.flags = (edge.flags | kSoftEdgeCutFlag | kSoftEdgeDisabledFlag) &
                             ~kSoftEdgeActiveFlag;
                CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
                CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
                return;
            }
        }
    }

    if (sampleDiagnostics)
    {
        InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kActiveAfterCutCounter),
                       1u, ignoredCounterValue);
    }
    CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
}
