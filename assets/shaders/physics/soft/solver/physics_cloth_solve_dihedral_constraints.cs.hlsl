#include "physics_particle_dispatch_constants.hlsli"
#include "physics_solver_config.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuClothDihedral, g_ClothDihedrals);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_BendLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBendCorrection, g_BendCorrections);

static void StoreZeroCorrection(uint correctionIndex)
{
    GpuBendCorrection correction;
    correction.correction0 = float4(0.0, 0.0, 0.0, 0.0);
    correction.correction1 = float4(0.0, 0.0, 0.0, 0.0);
    correction.correction2 = float4(0.0, 0.0, 0.0, 0.0);
    correction.correction3 = float4(0.0, 0.0, 0.0, 0.0);
    CRESSIM_SB_STORE(g_BendCorrections, correctionIndex, correction);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint dihedralIndex = dispatchThreadID.x;
    if (dihedralIndex >= clothDihedralCount)
    {
        return;
    }
    const uint correctionIndex = softBendCount + dihedralIndex;
    const GpuClothDihedral bend = CRESSIM_SB_LOAD(g_ClothDihedrals, dihedralIndex);
    const float4 edge0Mass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, bend.edgeParticle0);
    const float4 edge1Mass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, bend.edgeParticle1);
    const float4 opposite0Mass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, bend.oppositeParticle0);
    const float4 opposite1Mass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, bend.oppositeParticle1);
    const float3 p0 = opposite0Mass.xyz;
    const float3 p1 = opposite1Mass.xyz;
    const float3 p2 = edge0Mass.xyz;
    const float3 p3 = edge1Mass.xyz;
    const float3 edge = p3 - p2;
    const float edgeLengthSq = dot(edge, edge);
    const float3 rawNormal0 = cross(p2 - p0, p3 - p0);
    const float3 rawNormal1 = cross(p3 - p1, p2 - p1);
    const float normal0Sq = dot(rawNormal0, rawNormal0);
    const float normal1Sq = dot(rawNormal1, rawNormal1);
    if (edgeLengthSq <= kEpsilon || normal0Sq <= kEpsilon || normal1Sq <= kEpsilon)
    {
        StoreZeroCorrection(correctionIndex);
        return;
    }

    const float edgeLength = sqrt(edgeLengthSq);
    const float invEdgeLength = 1.0 / edgeLength;
    const float3 scaledNormal0 = rawNormal0 / normal0Sq;
    const float3 scaledNormal1 = rawNormal1 / normal1Sq;
    const float3 dOpposite0 = edgeLength * scaledNormal0;
    const float3 dOpposite1 = edgeLength * scaledNormal1;
    const float3 dEdge0 = dot(p0 - p3, edge) * invEdgeLength * scaledNormal0 +
                          dot(p1 - p3, edge) * invEdgeLength * scaledNormal1;
    const float3 dEdge1 = dot(p2 - p0, edge) * invEdgeLength * scaledNormal0 +
                          dot(p2 - p1, edge) * invEdgeLength * scaledNormal1;
    const float3 gradient0 = -dEdge0;
    const float3 gradient1 = -dEdge1;
    const float3 gradient2 = -dOpposite0;
    const float3 gradient3 = -dOpposite1;
    const float3 n0 = rawNormal0 / sqrt(normal0Sq);
    const float3 n1 = rawNormal1 / sqrt(normal1Sq);
    const float theta = atan2(dot(edge * invEdgeLength, cross(n0, n1)),
                              clamp(dot(n0, n1), -1.0, 1.0));
    const float twoPi = 6.283185307179586;
    float constraint = theta - bend.restAngle;
    constraint = constraint - twoPi * floor((constraint + 3.141592653589793) / twoPi);
    const float w0 = edge0Mass.w;
    const float w1 = edge1Mass.w;
    const float w2 = opposite0Mass.w;
    const float w3 = opposite1Mass.w;
    const float denominator = w0 * dot(gradient0, gradient0) +
                              w1 * dot(gradient1, gradient1) +
                              w2 * dot(gradient2, gradient2) +
                              w3 * dot(gradient3, gradient3);
    const float alpha = max(bend.compliance, 0.0) / max(dt * dt, kEpsilon);
    if (denominator + alpha <= kEpsilon)
    {
        StoreZeroCorrection(correctionIndex);
        return;
    }

    const float lambda = CRESSIM_SB_LOAD(g_BendLambdas, correctionIndex);
    const float deltaLambda = -(constraint + alpha * lambda) / (denominator + alpha);
    CRESSIM_SB_STORE(g_BendLambdas, correctionIndex, lambda + deltaLambda);
    GpuBendCorrection correction;
    correction.correction0 = float4(w0 * deltaLambda * gradient0 * kSoftInternalRelaxation, 0.0);
    correction.correction1 = float4(w1 * deltaLambda * gradient1 * kSoftInternalRelaxation, 0.0);
    correction.correction2 = float4(w2 * deltaLambda * gradient2 * kSoftInternalRelaxation, 0.0);
    correction.correction3 = float4(w3 * deltaLambda * gradient3 * kSoftInternalRelaxation, 0.0);
    CRESSIM_SB_STORE(g_BendCorrections, correctionIndex, correction);
}
