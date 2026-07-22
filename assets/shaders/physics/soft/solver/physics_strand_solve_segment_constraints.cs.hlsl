#include "physics_particle_dispatch_constants.hlsli"
#include "physics_solver_config.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_StrandSegmentLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandSegmentCorrection, g_StrandSegmentCorrections);

float3x3 OuterProduct3(float3 a, float3 b)
{
    return float3x3(a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y, a.y * b.z, a.z * b.x,
                    a.z * b.y, a.z * b.z);
}

float3 Mul3x3(float3x3 m, float3 v)
{
    return float3(dot(m[0], v), dot(m[1], v), dot(m[2], v));
}

float3x3 Inverse3x3(float3x3 m)
{
    const float a00 = m[0][0];
    const float a01 = m[0][1];
    const float a02 = m[0][2];
    const float a10 = m[1][0];
    const float a11 = m[1][1];
    const float a12 = m[1][2];
    const float a20 = m[2][0];
    const float a21 = m[2][1];
    const float a22 = m[2][2];
    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = a02 * a21 - a01 * a22;
    const float c02 = a01 * a12 - a02 * a11;
    const float c10 = a12 * a20 - a10 * a22;
    const float c11 = a00 * a22 - a02 * a20;
    const float c12 = a02 * a10 - a00 * a12;
    const float c20 = a10 * a21 - a11 * a20;
    const float c21 = a01 * a20 - a00 * a21;
    const float c22 = a00 * a11 - a01 * a10;
    const float det = a00 * c00 + a01 * c10 + a02 * c20;
    if (abs(det) <= kEpsilon)
    {
        return float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    const float invDet = rcp(det);
    return float3x3(c00 * invDet, c01 * invDet, c02 * invDet, c10 * invDet, c11 * invDet,
                    c12 * invDet, c20 * invDet, c21 * invDet, c22 * invDet);
}

float ComputeStrandOrientationInvMass(float wA, float wB, float restLength)
{
    const float dynamicWeightSum = max(wA + wB, 0.0);
    if (dynamicWeightSum <= kEpsilon || restLength <= kEpsilon)
    {
        return 0.0;
    }

    const float averageInvMass = dynamicWeightSum * 0.5;
    return 12.0 * averageInvMass / max(restLength * restLength, kEpsilon);
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
    correction.angularCorrection = float4(0.0, 0.0, 0.0, 0.0);

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

    const GpuStrandSegmentState state = CRESSIM_SB_LOAD(g_StrandSegmentStates, segmentIndex);
    const float4 q = QuaternionNormalize(state.orientation);
    const float3 axis = QuaternionRotate(q, float3(1.0, 0.0, 0.0));
    const float invRestLength = rcp(max(segment.restLength, kEpsilon));
    const float orientationInvMass =
        ComputeStrandOrientationInvMass(wA, wB, segment.restLength);
    const float3 constraint = delta * invRestLength - axis;
    const float alpha = max(segment.stretchShearCompliance, 0.0) / max(dt * dt, kEpsilon);
    const float4 lambdaState = CRESSIM_SB_LOAD(g_StrandSegmentLambdas, segmentIndex);
    const float3 lambda = lambdaState.xyz;

    const float3 qv = q.xyz;
    const float qw = q.w;
    // Paper gradient for c = delta / restLength - R(q) * e with local tangent e = +X.
    // The source expression is authored for quaternion order (w, x, y, z); we store (x, y, z, w),
    // so the scalar column is emitted last.
    const float3 gradQw = 2.0 * float3(qw, qv.z, -qv.y);
    const float3 gradQx = 2.0 * float3(0.0, qv.y, qv.z);
    const float3 gradQy = 2.0 * float3(-qv.y, 0.0, -qw);
    const float3 gradQz = 2.0 * float3(-qv.z, qw, 0.0);

    float3x3 system = float3x3(wSum * invRestLength * invRestLength + alpha, 0.0, 0.0, 0.0,
                               wSum * invRestLength * invRestLength + alpha, 0.0, 0.0, 0.0,
                               wSum * invRestLength * invRestLength + alpha);
    system += orientationInvMass * OuterProduct3(gradQx, gradQx);
    system += orientationInvMass * OuterProduct3(gradQy, gradQy);
    system += orientationInvMass * OuterProduct3(gradQz, gradQz);
    system += orientationInvMass * OuterProduct3(gradQw, gradQw);

    const float3 rhs = -(constraint + alpha * lambda);
    const float3 deltaLambda = Mul3x3(Inverse3x3(system), rhs);
    if (dot(deltaLambda, deltaLambda) > 0.0)
    {
        CRESSIM_SB_STORE(g_StrandSegmentLambdas, segmentIndex,
                         float4(lambda + deltaLambda, 0.0));
        correction.correctionA =
            float4(-wA * invRestLength * deltaLambda * kSoftInternalRelaxation, 0.0);
        correction.correctionB =
            float4(wB * invRestLength * deltaLambda * kSoftInternalRelaxation, 0.0);
        const float4 quatGradient = -float4(dot(gradQx, deltaLambda), dot(gradQy, deltaLambda),
                                            dot(gradQz, deltaLambda), dot(gradQw, deltaLambda));
        correction.angularCorrection =
            quatGradient * (orientationInvMass * kSoftInternalRelaxation);
    }

    CRESSIM_SB_STORE(g_StrandSegmentCorrections, segmentIndex, correction);
}
