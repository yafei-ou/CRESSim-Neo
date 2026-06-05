#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleStrandRoles);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidProxyLocalPositions);
CRESSIM_STRUCTURED_BUFFER(GpuStrandInsertionStateStorage, g_SuturingInsertionStates);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathHeader, g_SuturingPathHeaders);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathNode, g_SuturingPathNodes);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

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
    const float4 p0Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.x);
    const float4 p1Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.y);
    const float4 p2Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.z);
    const float4 p3Inv = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet.particleIndices.w);
    const uint ownerType = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleIndex);
    const bool proxy = ownerType == kParticleOwnerTypeRigidBody;

    float effectiveParticleMass = particlePosInvMass.w;
    float invMassRigid = 0.0;
    float3 invInertiaRigid = 0.0;
    float4 rigidOrientation = float4(0.0, 0.0, 0.0, 1.0);
    float3 rRigid = 0.0;
    uint rigidBodyIndex = 0u;
    if (proxy)
    {
        rigidBodyIndex = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleIndex);
        const uint bodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndex);
        const float4 rigidPositionInvMass =
            CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex);
        rigidOrientation =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
        invMassRigid = bodyType == kRigidBodyTypeDynamic ? rigidPositionInvMass.w : 0.0;
        invInertiaRigid = invMassRigid > kEpsilon
                              ? CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndex).xyz
                              : 0.0;
        const float3 localProxy = CRESSIM_SB_LOAD(g_RigidProxyLocalPositions, particleIndex).xyz;
        rRigid = QuaternionRotate(rigidOrientation, localProxy);
    }

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

    const float radialLength = sqrt(radialLengthSq);
    const float3 radialDirection = radial / radialLength;
    if (proxy)
    {
        effectiveParticleMass =
            ComputeContactEffectiveMass(invMassRigid, invInertiaRigid, rigidOrientation, rRigid,
                                        radialDirection);
    }

    const float effectiveTetMass =
        w0 * node.barycentrics.x * node.barycentrics.x +
        w1 * node.barycentrics.y * node.barycentrics.y +
        w2 * node.barycentrics.z * node.barycentrics.z +
        w3 * node.barycentrics.w * node.barycentrics.w;
    const float denom = effectiveParticleMass + effectiveTetMass;
    if (denom <= 1.0e-8)
    {
        return;
    }

    const float lambda = (radialLength / denom) * kSuturingRelaxation;
    float3 correction = radialDirection * lambda;
    const float correctionLength = length(correction);
    if (correctionLength > kSuturingMaxCorrection)
    {
        correction *= kSuturingMaxCorrection / max(correctionLength, kEpsilon);
    }

    if (!proxy && particlePosInvMass.w > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, particleIndex,
                                      -correction * particlePosInvMass.w);
    }

    if (proxy && invMassRigid > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, rigidBodyIndex,
                                      -correction * invMassRigid);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_RigidBodyRotationCorrections, rigidBodyIndex,
            -MultiplyWorldInverseInertia(invInertiaRigid, rigidOrientation,
                                         cross(rRigid, radialDirection)) *
                lambda);
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
