#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftEdgeCorrection, g_SoftEdgeCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint edgeIndex = dispatchThreadID.x;
    if (edgeIndex >= softEdgeCount)
    {
        return;
    }

    const GpuSoftEdge edge = CRESSIM_SB_LOAD(g_SoftEdges, edgeIndex);
    GpuSoftEdgeCorrection edgeCorrection;
    edgeCorrection.correctionA = float4(0.0, 0.0, 0.0, 0.0);
    edgeCorrection.correctionB = float4(0.0, 0.0, 0.0, 0.0);

    const uint particleA = edge.particleA;
    const uint particleB = edge.particleB;

    const float4 positionInvMassA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleA);
    const float4 positionInvMassB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleB);
    const float wA = positionInvMassA.w;
    const float wB = positionInvMassB.w;
    const float wSum = wA + wB;
    if (wSum <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
        return;
    }

    const float3 delta = positionInvMassB.xyz - positionInvMassA.xyz;
    const float lengthSq = dot(delta, delta);
    if (lengthSq <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_SoftEdgeCorrections, edgeIndex, edgeCorrection);
        return;
    }

    const float length = sqrt(lengthSq);
    const float3 gradientA = -delta / length;
    const float3 gradientB = -gradientA;
    const float compliance = max(edge.compliance, 0.0);
    const float alpha = compliance / max(dt * dt, kEpsilon);
    const float constraint = length - edge.restLength;
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
