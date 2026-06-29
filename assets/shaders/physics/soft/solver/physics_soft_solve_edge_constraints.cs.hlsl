#include "physics_particle_dispatch_constants.hlsli"
#include "physics_solver_config.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftEdgeCorrection, g_SoftEdgeCorrections);
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
    GpuSoftEdgeCorrection edgeCorrection;
    edgeCorrection.correctionA = float4(0.0, 0.0, 0.0, 0.0);
    edgeCorrection.correctionB = float4(0.0, 0.0, 0.0, 0.0);
    const bool sampleToolHits = cuttingToolEnabled != 0u && iterationIndex == 0u;
    const bool sampleToolDiagnostics =
        sampleToolHits && reserved1 > 0u && reserved0 + 1u == reserved1;
    uint ignoredCounterValue = 0u;

    if ((edge.flags & kSoftEdgeActiveFlag) == 0u ||
        (edge.flags & kSoftEdgeDisabledFlag) != 0u)
    {
        if (sampleToolDiagnostics)
        {
            InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kAlreadyDisabledCounter),
                           1u, ignoredCounterValue);
        }
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
        CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
        return;
    }

    const uint particleA = edge.particleA;
    const uint particleB = edge.particleB;

    const float4 positionInvMassA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleA);
    const float4 positionInvMassB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleB);
    const float wA = positionInvMassA.w;
    const float wB = positionInvMassB.w;
    const float wSum = wA + wB;
    if (wSum <= kEpsilon)
    {
        if (sampleToolDiagnostics)
        {
            InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kActiveAfterCutCounter),
                           1u, ignoredCounterValue);
        }
        CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
        return;
    }

    const float3 delta = positionInvMassB.xyz - positionInvMassA.xyz;
    const float lengthSq = dot(delta, delta);
    if (lengthSq <= kEpsilon)
    {
        if (sampleToolDiagnostics)
        {
            InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kActiveAfterCutCounter),
                           1u, ignoredCounterValue);
        }
        CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
        return;
    }

    const float length = sqrt(lengthSq);
    const float3 gradientA = -delta / length;
    const float3 gradientB = -gradientA;
    const float compliance = max(edge.compliance, 0.0);
    const float alpha = compliance / max(dt * dt, kEpsilon);
    const float constraint = length - edge.restLength;

    if (cuttingToolEnabled != 0u && cuttingToolRadius > 0.0 &&
        cuttingToolStrength > 0.0)
    {
        const float toolDistance =
            segmentSegmentDistance(positionInvMassA.xyz, positionInvMassB.xyz,
                                   cuttingToolTipA, cuttingToolTipB);
        if (toolDistance <= cuttingToolRadius)
        {
            if (sampleToolHits)
            {
                InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kToolCandidateCounter),
                               1u, ignoredCounterValue);
            }
            if (iterationIndex == 0u)
            {
                edge.damage = saturate(edge.damage + cuttingToolStrength * dt);
            }
            if (edge.damage >= edge.cutResistance)
            {
                if (iterationIndex == 0u)
                {
                    InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kNewlyCutCounter),
                                   1u, ignoredCounterValue);
                }
                edge.flags = (edge.flags | kSoftEdgeCutFlag | kSoftEdgeDisabledFlag) &
                             ~kSoftEdgeActiveFlag;
                CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
                CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
                CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
                return;
            }
        }
    }

    const float strainDenominator = max(edge.restLength, kEpsilon);
    edge.strain = constraint / strainDenominator;
    const float strainMagnitude = abs(edge.strain);
    if (edge.failureThreshold > 0.0 && strainMagnitude > edge.failureThreshold)
    {
        const float excess = strainMagnitude - edge.failureThreshold;
        edge.damage = saturate(edge.damage + excess / max(edge.cutResistance, kEpsilon));
        if (edge.damage >= 1.0)
        {
            edge.flags = (edge.flags | kSoftEdgeFracturedFlag | kSoftEdgeDisabledFlag) &
                         ~kSoftEdgeActiveFlag;
            CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
            CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
            CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
            return;
        }
    }
    CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);

    if (sampleToolDiagnostics)
    {
        InterlockedAdd(CRESSIM_SB_REF(g_SoftEdgeToolCounters, kActiveAfterCutCounter),
                       1u, ignoredCounterValue);
    }

    const float lambda = CRESSIM_SB_LOAD(g_SoftEdgeLambdas, edgeIndex);
    const float denominator = wSum + alpha;
    if (denominator <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
        return;
    }

    const float deltaLambda = -(constraint + alpha * lambda) / denominator;
    CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, lambda + deltaLambda);

    const float3 correctionA = wA * deltaLambda * gradientA * kSoftInternalRelaxation;
    const float3 correctionB = wB * deltaLambda * gradientB * kSoftInternalRelaxation;
    edgeCorrection.correctionA = float4(correctionA, 0.0);
    edgeCorrection.correctionB = float4(correctionB, 0.0);
    CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
}
