#include "physics_particle_dispatch_constants.hlsli"
#include "physics_solver_config.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuStrandJoint, g_StrandJoints);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_StrandJointLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandJointCorrection, g_StrandJointCorrections);

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
    const uint jointIndex = dispatchThreadID.x;
    if (jointIndex >= strandJointCount)
    {
        return;
    }

    const GpuStrandJoint joint = CRESSIM_SB_LOAD(g_StrandJoints, jointIndex);
    const GpuStrandSegment segmentA = CRESSIM_SB_LOAD(g_StrandSegments, joint.segmentA);
    const GpuStrandSegment segmentB = CRESSIM_SB_LOAD(g_StrandSegments, joint.segmentB);
    GpuStrandJointCorrection correction;
    correction.correction0 = float4(0.0, 0.0, 0.0, 0.0);
    correction.correction1 = float4(0.0, 0.0, 0.0, 0.0);
    correction.correction2 = float4(0.0, 0.0, 0.0, 0.0);
    correction.twistRotationA = float4(0.0, 0.0, 0.0, 0.0);
    correction.twistRotationB = float4(0.0, 0.0, 0.0, 0.0);
    const float4 orientationA = QuaternionNormalize(
        CRESSIM_SB_LOAD(g_StrandSegmentStates, joint.segmentA).orientation);
    const float4 orientationB = QuaternionNormalize(
        CRESSIM_SB_LOAD(g_StrandSegmentStates, joint.segmentB).orientation);
    const float wA0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentA.particleA).w;
    const float wA1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentA.particleB).w;
    const float wB0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentB.particleA).w;
    const float wB1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segmentB.particleB).w;
    const float orientationInvMassA =
        ComputeStrandOrientationInvMass(wA0, wA1, segmentA.restLength);
    const float orientationInvMassB =
        ComputeStrandOrientationInvMass(wB0, wB1, segmentB.restLength);
    float4 relative = QuaternionNormalize(QuaternionMul(QuaternionConjugate(orientationA), orientationB));
    float4 restRelative = QuaternionNormalize(joint.restRelativeOrientation);
    if (dot(relative, restRelative) < 0.0)
    {
        restRelative = -restRelative;
    }

    const float3 constraint = relative.xyz - restRelative.xyz;
    const float4 lambda = CRESSIM_SB_LOAD(g_StrandJointLambdas, jointIndex);
    const float bendAlpha = max(joint.bendCompliance, 0.0) / max(dt * dt, kEpsilon);
    const float twistAlpha = max(joint.twistCompliance, 0.0) / max(dt * dt, kEpsilon);
    const float3 alphaVector = float3(twistAlpha, bendAlpha, bendAlpha);
    const float3x3 alphaMatrix = float3x3(alphaVector.x, 0.0, 0.0, 0.0, alphaVector.y, 0.0, 0.0,
                                          0.0, alphaVector.z);

    const float3 va = orientationA.xyz;
    const float wa = orientationA.w;
    const float3 vb = orientationB.xyz;
    const float wb = orientationB.w;

    const float3 gradAX = float3(-wb, vb.z, -vb.y);
    const float3 gradAY = float3(-vb.z, -wb, vb.x);
    const float3 gradAZ = float3(vb.y, -vb.x, -wb);
    const float3 gradAW = vb;

    const float3 gradBX = float3(wa, -va.z, va.y);
    const float3 gradBY = float3(va.z, wa, -va.x);
    const float3 gradBZ = float3(-va.y, va.x, wa);
    const float3 gradBW = -va;

    float3x3 system = alphaMatrix;
    system += orientationInvMassA * OuterProduct3(gradAX, gradAX);
    system += orientationInvMassA * OuterProduct3(gradAY, gradAY);
    system += orientationInvMassA * OuterProduct3(gradAZ, gradAZ);
    system += orientationInvMassA * OuterProduct3(gradAW, gradAW);
    system += orientationInvMassB * OuterProduct3(gradBX, gradBX);
    system += orientationInvMassB * OuterProduct3(gradBY, gradBY);
    system += orientationInvMassB * OuterProduct3(gradBZ, gradBZ);
    system += orientationInvMassB * OuterProduct3(gradBW, gradBW);

    const float3 rhs = -(constraint + alphaVector * lambda.xyz);
    const float3 deltaLambda = Mul3x3(Inverse3x3(system), rhs);
    if (dot(deltaLambda, deltaLambda) > 0.0)
    {
        CRESSIM_SB_STORE(g_StrandJointLambdas, jointIndex,
                         float4(lambda.xyz + deltaLambda, 0.0));
        correction.twistRotationA =
            float4(dot(gradAX, deltaLambda), dot(gradAY, deltaLambda), dot(gradAZ, deltaLambda),
                   dot(gradAW, deltaLambda)) *
            (orientationInvMassA * kSoftInternalRelaxation);
        correction.twistRotationB =
            float4(dot(gradBX, deltaLambda), dot(gradBY, deltaLambda), dot(gradBZ, deltaLambda),
                   dot(gradBW, deltaLambda)) *
            (orientationInvMassB * kSoftInternalRelaxation);
    }

    CRESSIM_SB_STORE(g_StrandJointCorrections, jointIndex, correction);
}
