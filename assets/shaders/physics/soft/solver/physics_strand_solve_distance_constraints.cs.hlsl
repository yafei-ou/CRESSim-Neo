#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuStrandDistanceConstraint, g_StrandDistanceConstraints);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_StrandDistanceLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftEdgeCorrection, g_StrandDistanceCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint constraintIndex = dispatchThreadID.x;
    if (constraintIndex >= strandDistanceCount)
    {
        return;
    }

    const GpuStrandDistanceConstraint constraint =
        CRESSIM_SB_LOAD(g_StrandDistanceConstraints, constraintIndex);
    GpuSoftEdgeCorrection correction;
    correction.correctionA = float4(0.0, 0.0, 0.0, 0.0);
    correction.correctionB = float4(0.0, 0.0, 0.0, 0.0);

    const float4 positionInvMassA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, constraint.particleA);
    const float4 positionInvMassB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, constraint.particleB);
    const float wA = positionInvMassA.w;
    const float wB = positionInvMassB.w;
    const float wSum = wA + wB;
    if (wSum <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_StrandDistanceCorrections, constraintIndex, correction);
        return;
    }

    const float3 delta = positionInvMassB.xyz - positionInvMassA.xyz;
    const float lengthSq = dot(delta, delta);
    if (lengthSq <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_StrandDistanceCorrections, constraintIndex, correction);
        return;
    }

    const float length = sqrt(lengthSq);
    const float3 gradient = delta / length;
    const float alpha = max(constraint.distanceCompliance, 0.0) / max(dt * dt, kEpsilon);
    const float lambda = CRESSIM_SB_LOAD(g_StrandDistanceLambdas, constraintIndex);
    const float constraintValue = length - constraint.restLength;
    const float denominator = wSum + alpha;
    if (denominator > kEpsilon)
    {
        const float deltaLambda = -(constraintValue + alpha * lambda) / denominator;
        CRESSIM_SB_STORE(g_StrandDistanceLambdas, constraintIndex, lambda + deltaLambda);
        correction.correctionA = float4(wA * deltaLambda * -gradient * kSoftInternalRelaxation, 0.0);
        correction.correctionB = float4(wB * deltaLambda * gradient * kSoftInternalRelaxation, 0.0);
    }

    CRESSIM_SB_STORE(g_StrandDistanceCorrections, constraintIndex, correction);
}
