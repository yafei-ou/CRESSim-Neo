#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_StrandSegmentLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandSegmentCorrection, g_StrandSegmentCorrections);

float4 ComputeSegmentOrientation(float3 delta, float3 materialHint)
{
    const float3 tangent = SafeNormalize(delta, float3(1.0, 0.0, 0.0));
    float3 normal = materialHint.xyz - tangent * dot(materialHint.xyz, tangent);
    if (dot(normal, normal) <= 1.0e-8)
    {
        normal = abs(tangent.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        normal -= tangent * dot(normal, tangent);
    }
    normal = SafeNormalize(normal, float3(0.0, 1.0, 0.0));
    const float3 binormal = SafeNormalize(cross(tangent, normal), float3(0.0, 0.0, 1.0));

    const float m00 = tangent.x;
    const float m11 = normal.y;
    const float m22 = binormal.z;
    const float trace = m00 + m11 + m22;
    float4 q;
    if (trace > 0.0)
    {
        const float s = sqrt(trace + 1.0) * 2.0;
        q = float4((normal.z - binormal.y) / s, (binormal.x - tangent.z) / s,
                   (tangent.y - normal.x) / s, 0.25 * s);
    }
    else if (tangent.x > normal.y && tangent.x > binormal.z)
    {
        const float s = sqrt(1.0 + tangent.x - normal.y - binormal.z) * 2.0;
        q = float4(0.25 * s, (normal.x + tangent.y) / s, (binormal.x + tangent.z) / s,
                   (normal.z - binormal.y) / s);
    }
    else if (normal.y > binormal.z)
    {
        const float s = sqrt(1.0 + normal.y - tangent.x - binormal.z) * 2.0;
        q = float4((normal.x + tangent.y) / s, 0.25 * s, (binormal.y + normal.z) / s,
                   (binormal.x - tangent.z) / s);
    }
    else
    {
        const float s = sqrt(1.0 + binormal.z - tangent.x - normal.y) * 2.0;
        q = float4((binormal.x + tangent.z) / s, (binormal.y + normal.z) / s, 0.25 * s,
                   (tangent.y - normal.x) / s);
    }
    return QuaternionNormalize(q);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint segmentIndex = dispatchThreadID.x;
    if (segmentIndex >= strandSegmentCount)
    {
        return;
    }

    const GpuStrandSegment segment = CRESSIM_SB_LOAD(g_StrandSegments, segmentIndex);
    const float4 positionInvMassA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleA);
    const float4 positionInvMassB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleB);
    const float wA = positionInvMassA.w;
    const float wB = positionInvMassB.w;
    const float wSum = wA + wB;

    GpuStrandSegmentCorrection correction;
    correction.correctionA = float4(0.0, 0.0, 0.0, 0.0);
    correction.correctionB = float4(0.0, 0.0, 0.0, 0.0);
    correction.orientation = segment.restOrientation;

    if (wSum <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_StrandSegmentCorrections, segmentIndex, correction);
        return;
    }

    const float3 delta = positionInvMassB.xyz - positionInvMassA.xyz;
    const float lengthSq = dot(delta, delta);
    if (lengthSq <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_StrandSegmentCorrections, segmentIndex, correction);
        return;
    }

    const float length = sqrt(lengthSq);
    const float3 gradientA = -delta / length;
    const float3 gradientB = -gradientA;
    const float alpha = max(segment.compliance, 0.0) / max(dt * dt, kEpsilon);
    const float constraint = length - segment.restLength;
    const float lambda = CRESSIM_SB_LOAD(g_StrandSegmentLambdas, segmentIndex);
    const float denominator = wSum + alpha;
    if (denominator > kEpsilon)
    {
        const float deltaLambda = -(constraint + alpha * lambda) / denominator;
        CRESSIM_SB_STORE(g_StrandSegmentLambdas, segmentIndex, lambda + deltaLambda);
        correction.correctionA =
            float4(wA * deltaLambda * gradientA * kSoftInternalRelaxation, 0.0);
        correction.correctionB =
            float4(wB * deltaLambda * gradientB * kSoftInternalRelaxation, 0.0);
    }

    correction.orientation = ComputeSegmentOrientation(delta, segment.materialFrame);
    CRESSIM_SB_STORE(g_StrandSegmentCorrections, segmentIndex, correction);
}
