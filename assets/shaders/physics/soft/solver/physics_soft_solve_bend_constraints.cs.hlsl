#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftBend, g_SoftBends);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftBendLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftBendCorrection, g_SoftBendCorrections);

static void StoreZeroCorrection(uint bendIndex)
{
    GpuSoftBendCorrection bendCorrection;
    bendCorrection.correction0 = float4(0.0, 0.0, 0.0, 0.0);
    bendCorrection.correction1 = float4(0.0, 0.0, 0.0, 0.0);
    bendCorrection.correction2 = float4(0.0, 0.0, 0.0, 0.0);
    CRESSIM_SB_STORE(g_SoftBendCorrections, bendIndex, bendCorrection);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bendIndex = dispatchThreadID.x;
    if (bendIndex >= softBendCount)
    {
        return;
    }

    const GpuSoftBend bend = CRESSIM_SB_LOAD(g_SoftBends, bendIndex);
    const float4 positionInvMass0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, bend.particle0);
    const float4 positionInvMass1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, bend.particle1);
    const float4 positionInvMass2 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, bend.particle2);

    const float w0 = positionInvMass0.w;
    const float w1 = positionInvMass1.w;
    const float w2 = positionInvMass2.w;
    if (w0 + w1 + w2 <= kEpsilon)
    {
        StoreZeroCorrection(bendIndex);
        return;
    }

    const float3 edge0 = positionInvMass0.xyz - positionInvMass1.xyz;
    const float3 edge1 = positionInvMass2.xyz - positionInvMass1.xyz;
    const float length0Sq = dot(edge0, edge0);
    const float length1Sq = dot(edge1, edge1);
    if (length0Sq <= kEpsilon || length1Sq <= kEpsilon)
    {
        StoreZeroCorrection(bendIndex);
        return;
    }

    const float length0 = sqrt(length0Sq);
    const float length1 = sqrt(length1Sq);
    const float3 dir0 = edge0 / length0;
    const float3 dir1 = edge1 / length1;
    const float cosTheta = clamp(dot(dir0, dir1), -1.0 + 1.0e-4, 1.0 - 1.0e-4);
    const float sinThetaSq = max(1.0 - cosTheta * cosTheta, 1.0e-8);
    const float sinTheta = sqrt(sinThetaSq);
    const float theta = acos(cosTheta);

    const float3 gradient0 = -(dir1 - cosTheta * dir0) / max(length0 * sinTheta, kEpsilon);
    const float3 gradient2 = -(dir0 - cosTheta * dir1) / max(length1 * sinTheta, kEpsilon);
    const float3 gradient1 = -(gradient0 + gradient2);

    const float denominator =
        w0 * dot(gradient0, gradient0) + w1 * dot(gradient1, gradient1) +
        w2 * dot(gradient2, gradient2);
    const float compliance = max(bend.compliance, 0.0);
    const float alpha = compliance / max(dt * dt, kEpsilon);
    if (denominator + alpha <= kEpsilon)
    {
        StoreZeroCorrection(bendIndex);
        return;
    }

    const float constraint = theta - bend.restAngle;
    const float lambda = CRESSIM_SB_LOAD(g_SoftBendLambdas, bendIndex);
    const float deltaLambda = -(constraint + alpha * lambda) / (denominator + alpha);
    CRESSIM_SB_STORE(g_SoftBendLambdas, bendIndex, lambda + deltaLambda);

    GpuSoftBendCorrection bendCorrection;
    bendCorrection.correction0 = float4(w0 * deltaLambda * gradient0 * kSoftInternalRelaxation, 0.0);
    bendCorrection.correction1 = float4(w1 * deltaLambda * gradient1 * kSoftInternalRelaxation, 0.0);
    bendCorrection.correction2 = float4(w2 * deltaLambda * gradient2 * kSoftInternalRelaxation, 0.0);
    CRESSIM_SB_STORE(g_SoftBendCorrections, bendIndex, bendCorrection);
}
