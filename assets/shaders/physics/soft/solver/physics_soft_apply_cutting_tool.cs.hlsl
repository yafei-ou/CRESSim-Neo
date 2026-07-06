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

    const bool toolEnabled = cuttingToolEnabled != 0u && cuttingToolRadius > 0.0 &&
                             cuttingToolStrength > 0.0;
    if (toolEnabled)
    {
        const float3 positionA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleA).xyz;
        const float3 positionB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleB).xyz;
        const float toolDistance =
            segmentSegmentDistance(positionA, positionB, cuttingToolTipA, cuttingToolTipB);
        if (toolDistance <= cuttingToolRadius)
        {
            InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kToolCandidateCounter),
                           1u, ignoredCounterValue);
            edge.damage = saturate(edge.damage + cuttingToolStrength * dt);
            if (edge.damage >= edge.cutResistance)
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
