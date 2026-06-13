#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_STRUCTURED_BUFFER(GpuStrandJoint, g_StrandJoints);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_StrandJointLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandJointCorrection, g_StrandJointCorrections);

float ComputeTwistConstraint(float4 qA, float4 qB, float4 restRelative)
{
    const float4 relative = QuaternionMul(QuaternionConjugate(QuaternionNormalize(qA)),
                                          QuaternionNormalize(qB));
    const float4 errorQ =
        QuaternionMul(QuaternionConjugate(QuaternionNormalize(restRelative)), relative);
    return 2.0 * acos(clamp(errorQ.w, -1.0, 1.0));
}

float ComputeRestBendAngle(float4 restRelativeOrientation)
{
    const float3 restTangentB = QuaternionRotate(QuaternionNormalize(restRelativeOrientation),
                                                 float3(1.0, 0.0, 0.0));
    const float restCosTheta = clamp(dot(float3(-1.0, 0.0, 0.0), restTangentB),
                                     -1.0 + 1.0e-4, 1.0 - 1.0e-4);
    return acos(restCosTheta);
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
    const uint particle0 = segmentA.particleA;
    const uint particle1 = segmentA.particleB;
    const uint particle2 = segmentB.particleB;

    const float4 positionInvMass0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particle0);
    const float4 positionInvMass1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particle1);
    const float4 positionInvMass2 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particle2);

    GpuStrandJointCorrection correction;
    correction.correction0 = float4(0.0, 0.0, 0.0, 0.0);
    correction.correction1 = float4(0.0, 0.0, 0.0, 0.0);
    correction.correction2 = float4(0.0, 0.0, 0.0, 0.0);
    correction.orientationA = CRESSIM_SB_LOAD(g_StrandSegmentStates, joint.segmentA).orientation;
    correction.orientationB = CRESSIM_SB_LOAD(g_StrandSegmentStates, joint.segmentB).orientation;

    const float w0 = positionInvMass0.w;
    const float w1 = positionInvMass1.w;
    const float w2 = positionInvMass2.w;
    if (w0 + w1 + w2 <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_StrandJointCorrections, jointIndex, correction);
        return;
    }

    const float3 edge0 = positionInvMass0.xyz - positionInvMass1.xyz;
    const float3 edge1 = positionInvMass2.xyz - positionInvMass1.xyz;
    const float length0Sq = dot(edge0, edge0);
    const float length1Sq = dot(edge1, edge1);
    if (length0Sq <= kEpsilon || length1Sq <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_StrandJointCorrections, jointIndex, correction);
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

    const float bendAlpha = max(joint.bendCompliance, 0.0) / max(dt * dt, kEpsilon);
    const float restAngle = ComputeRestBendAngle(joint.restRelativeOrientation);
    const float constraint = theta - restAngle;
    const float lambda = CRESSIM_SB_LOAD(g_StrandJointLambdas, jointIndex);
    if (denominator + bendAlpha > kEpsilon)
    {
        const float deltaLambda =
            -(constraint + bendAlpha * lambda) / (denominator + bendAlpha);
        CRESSIM_SB_STORE(g_StrandJointLambdas, jointIndex, lambda + deltaLambda);
        correction.correction0 = float4(w0 * deltaLambda * gradient0 * kSoftInternalRelaxation, 0.0);
        correction.correction1 = float4(w1 * deltaLambda * gradient1 * kSoftInternalRelaxation, 0.0);
        correction.correction2 = float4(w2 * deltaLambda * gradient2 * kSoftInternalRelaxation, 0.0);
    }

    CRESSIM_SB_STORE(g_StrandJointCorrections, jointIndex, correction);
}
