#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

static const float kSoftCorrectionAtomicScale = 100000.0;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftPositionCorrections);

int3 QuantizeSoftCorrection(float3 value)
{
    return int3(round(value * kSoftCorrectionAtomicScale));
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint edgeIndex = dispatchThreadID.x;
    if (edgeIndex >= softEdgeCount)
    {
        return;
    }

    const GpuSoftEdge edge = CRESSIM_SB_LOAD(g_SoftEdges, edgeIndex);
    const uint particleA = edge.particleA;
    const uint particleB = edge.particleB;

    const float4 positionInvMassA = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleA);
    const float4 positionInvMassB = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleB);
    const float wA = positionInvMassA.w;
    const float wB = positionInvMassB.w;
    const float wSum = wA + wB;
    if (wSum <= kEpsilon)
    {
        return;
    }

    const float3 delta = positionInvMassB.xyz - positionInvMassA.xyz;
    const float lengthSq = dot(delta, delta);
    if (lengthSq <= kEpsilon)
    {
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
        return;
    }

    const float deltaLambda = -(constraint + alpha * lambda) / denominator;
    CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, lambda + deltaLambda);

    const float3 correctionA = wA * deltaLambda * gradientA * kSoftInternalRelaxation;
    const float3 correctionB = wB * deltaLambda * gradientB * kSoftInternalRelaxation;
    const int3 quantizedA = QuantizeSoftCorrection(correctionA);
    const int3 quantizedB = QuantizeSoftCorrection(correctionB);

    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, particleA).x, quantizedA.x);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, particleA).y, quantizedA.y);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, particleA).z, quantizedA.z);

    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, particleB).x, quantizedB.x);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, particleB).y, quantizedB.y);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, particleB).z, quantizedB.z);
}
