#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

static const float kSoftCorrectionAtomicScale = 100000.0;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftTetLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftPositionCorrections);

int3 QuantizeSoftCorrection(float3 value)
{
    return int3(round(value * kSoftCorrectionAtomicScale));
}

float SignedTetVolume(float3 p0, float3 p1, float3 p2, float3 p3)
{
    return dot(cross(p1 - p0, p2 - p0), p3 - p0) / 6.0;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint tetIndex = dispatchThreadID.x;
    if (tetIndex >= softTetCount)
    {
        return;
    }

    const GpuSoftTet tet = CRESSIM_SB_LOAD(g_SoftTets, tetIndex);
    const uint i0 = tet.particleIndices.x;
    const uint i1 = tet.particleIndices.y;
    const uint i2 = tet.particleIndices.z;
    const uint i3 = tet.particleIndices.w;

    const float4 p0Inv = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, i0);
    const float4 p1Inv = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, i1);
    const float4 p2Inv = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, i2);
    const float4 p3Inv = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, i3);

    const float w0 = p0Inv.w;
    const float w1 = p1Inv.w;
    const float w2 = p2Inv.w;
    const float w3 = p3Inv.w;
    if ((w0 + w1 + w2 + w3) <= kEpsilon)
    {
        return;
    }

    const float3 p0 = p0Inv.xyz;
    const float3 p1 = p1Inv.xyz;
    const float3 p2 = p2Inv.xyz;
    const float3 p3 = p3Inv.xyz;

    const float signedVolume = SignedTetVolume(p0, p1, p2, p3);
    const float volumeSign = (signedVolume >= 0.0) ? 1.0 : -1.0;
    const float volume = abs(signedVolume);
    const float restVolume = max(tet.restVolume, 0.0);
    const float constraint = volume - restVolume;

    const float3 gradient0 = cross(p3 - p1, p2 - p1) * (volumeSign / 6.0);
    const float3 gradient1 = cross(p2 - p0, p3 - p0) * (volumeSign / 6.0);
    const float3 gradient2 = cross(p3 - p0, p1 - p0) * (volumeSign / 6.0);
    const float3 gradient3 = cross(p1 - p0, p2 - p0) * (volumeSign / 6.0);

    const float sumGradSq = w0 * dot(gradient0, gradient0) +
                            w1 * dot(gradient1, gradient1) +
                            w2 * dot(gradient2, gradient2) +
                            w3 * dot(gradient3, gradient3);
    const float compliance = max(tet.compliance, 0.0);
    const float alpha = compliance / max(dt * dt, kEpsilon);
    const float lambda = CRESSIM_SB_LOAD(g_SoftTetLambdas, tetIndex);
    const float denominator = sumGradSq + alpha;
    if (denominator <= kEpsilon)
    {
        return;
    }

    const float deltaLambda = -(constraint + alpha * lambda) / denominator;
    CRESSIM_SB_STORE(g_SoftTetLambdas, tetIndex, lambda + deltaLambda);

    const float3 correction0 = w0 * deltaLambda * gradient0 * kSoftInternalRelaxation;
    const float3 correction1 = w1 * deltaLambda * gradient1 * kSoftInternalRelaxation;
    const float3 correction2 = w2 * deltaLambda * gradient2 * kSoftInternalRelaxation;
    const float3 correction3 = w3 * deltaLambda * gradient3 * kSoftInternalRelaxation;

    const int3 quantized0 = QuantizeSoftCorrection(correction0);
    const int3 quantized1 = QuantizeSoftCorrection(correction1);
    const int3 quantized2 = QuantizeSoftCorrection(correction2);
    const int3 quantized3 = QuantizeSoftCorrection(correction3);

    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i0).x, quantized0.x);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i0).y, quantized0.y);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i0).z, quantized0.z);

    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i1).x, quantized1.x);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i1).y, quantized1.y);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i1).z, quantized1.z);

    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i2).x, quantized2.x);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i2).y, quantized2.y);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i2).z, quantized2.z);

    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i3).x, quantized3.x);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i3).y, quantized3.y);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, i3).z, quantized3.z);
}
