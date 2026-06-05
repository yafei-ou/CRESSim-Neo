#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleStrandRoles);
CRESSIM_STRUCTURED_BUFFER(GpuStrandInsertionStateStorage, g_SuturingInsertionStates);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathHeader, g_SuturingPathHeaders);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathNode, g_SuturingPathNodes);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);

static const float kSuturingRelaxation = 0.05;
static const float kSuturingMaxCorrection = 0.02;

float3 EvaluatePathNodePosition(GpuSuturingPathNode node)
{
    if (node.tetIndex == kInvalidSuturingIndex || node.tetIndex >= softTetCount)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const GpuSoftTet tet = CRESSIM_SB_LOAD(g_SoftTets, node.tetIndex);
    const float3 p0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.x).xyz;
    const float3 p1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.y).xyz;
    const float3 p2 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.z).xyz;
    const float3 p3 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.w).xyz;
    return p0 * node.barycentrics.x + p1 * node.barycentrics.y + p2 * node.barycentrics.z +
           p3 * node.barycentrics.w;
}

float3 EvaluatePathTangent(uint pathIndex, uint nodeIndex)
{
    if (pathIndex == kInvalidSuturingIndex || pathIndex >= suturingPathHeaderCount ||
        nodeIndex == kInvalidSuturingIndex || nodeIndex >= suturingPathNodeCount)
    {
        return float3(0.0, 0.0, 1.0);
    }

    const GpuSuturingPathHeader header = CRESSIM_SB_LOAD(g_SuturingPathHeaders, pathIndex);
    if (header.nodeCount == 0u || nodeIndex < header.nodeStart ||
        nodeIndex >= header.nodeStart + header.nodeCount)
    {
        return float3(0.0, 0.0, 1.0);
    }

    float3 tangent = float3(0.0, 0.0, 0.0);
    const float3 currentPosition =
        EvaluatePathNodePosition(CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex));
    if (nodeIndex > header.nodeStart)
    {
        const float3 prevPosition =
            EvaluatePathNodePosition(CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex - 1u));
        tangent += currentPosition - prevPosition;
    }
    if (nodeIndex + 1u < header.nodeStart + header.nodeCount)
    {
        const float3 nextPosition =
            EvaluatePathNodePosition(CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex + 1u));
        tangent += nextPosition - currentPosition;
    }

    if (dot(tangent, tangent) > kEpsilon)
    {
        return normalize(tangent);
    }

    const GpuSuturingPathNode node = CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex);
    tangent = node.tangentArcLength.xyz;
    if (dot(tangent, tangent) > kEpsilon)
    {
        return normalize(tangent);
    }

    return float3(0.0, 0.0, 1.0);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    const uint strandRole = CRESSIM_SB_LOAD(g_ParticleStrandRoles, particleIndex);
    if (strandRole == kParticleStrandRoleNone || strandRole == kParticleStrandRoleNeedleTip)
    {
        return;
    }

    const GpuStrandInsertionStateStorage state = CRESSIM_SB_LOAD(g_SuturingInsertionStates, particleIndex);
    if (state.state != kStrandInsertionStateInside || state.pathIndex == kInvalidSuturingIndex ||
        state.pathIndex >= suturingPathHeaderCount ||
        state.nearestNodeIndex == kInvalidSuturingIndex ||
        state.nearestNodeIndex >= suturingPathNodeCount)
    {
        return;
    }

    const GpuSuturingPathNode node = CRESSIM_SB_LOAD(g_SuturingPathNodes, state.nearestNodeIndex);
    if (node.tetIndex == kInvalidSuturingIndex || node.tetIndex >= softTetCount)
    {
        return;
    }

    const GpuSoftTet tet = CRESSIM_SB_LOAD(g_SoftTets, node.tetIndex);
    const float4 particlePosInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float invMassParticle = particlePosInvMass.w;
    const float4 p0Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.x);
    const float4 p1Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.y);
    const float4 p2Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.z);
    const float4 p3Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.w);

    const float w0 = p0Inv.w;
    const float w1 = p1Inv.w;
    const float w2 = p2Inv.w;
    const float w3 = p3Inv.w;

    const float3 pathPosition = EvaluatePathNodePosition(node);
    const float3 tangent = EvaluatePathTangent(state.pathIndex, state.nearestNodeIndex);
    const float3 delta = particlePosInvMass.xyz - pathPosition;
    const float3 radial = delta - tangent * dot(delta, tangent);
    const float radialLengthSq = dot(radial, radial);
    if (radialLengthSq <= 1.0e-8)
    {
        return;
    }

    const float effectiveTetMass =
        w0 * node.barycentrics.x * node.barycentrics.x +
        w1 * node.barycentrics.y * node.barycentrics.y +
        w2 * node.barycentrics.z * node.barycentrics.z +
        w3 * node.barycentrics.w * node.barycentrics.w;
    const float denom = invMassParticle + effectiveTetMass;
    if (denom <= 1.0e-8)
    {
        return;
    }

    float3 correction = (radial / denom) * kSuturingRelaxation;
    const float correctionLength = length(correction);
    if (correctionLength > kSuturingMaxCorrection)
    {
        correction *= kSuturingMaxCorrection / max(correctionLength, kEpsilon);
    }

    if (invMassParticle > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, particleIndex,
                                      -correction * invMassParticle);
    }

    if (w0 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, tet.particleIndices.x,
                                      correction * w0 * node.barycentrics.x);
    }
    if (w1 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, tet.particleIndices.y,
                                      correction * w1 * node.barycentrics.y);
    }
    if (w2 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, tet.particleIndices.z,
                                      correction * w2 * node.barycentrics.z);
    }
    if (w3 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, tet.particleIndices.w,
                                      correction * w3 * node.barycentrics.w);
    }
}
