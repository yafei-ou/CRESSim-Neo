#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/physics_atomic_float.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "physics/rigid/physics_rigid_solver_shared.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint4, g_SuturingParticleRefs);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidProxyLocalPositions);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingInsertionStateStorage, g_SuturingInsertionStates);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathHeader, g_SuturingPathHeaders);
CRESSIM_STRUCTURED_BUFFER(GpuSuturingPathNode, g_SuturingPathNodes);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTet, g_SoftTets);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

static const float kSuturingRelaxation = 0.2;
static const float kSuturingMaxCorrection = 0.2;

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
        nodeIndex == kInvalidSuturingIndex || nodeIndex >= suturingPathNodeCount ||
        maxSuturingNodesPerPath == 0u)
    {
        return float3(0.0, 0.0, 1.0);
    }

    const GpuSuturingPathNode node = CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex);
    const uint nodeWindowStart = pathIndex * maxSuturingNodesPerPath;
    const uint nodeWindowEnd = min(nodeWindowStart + maxSuturingNodesPerPath, suturingPathNodeCount);
    if (nodeIndex < nodeWindowStart || nodeIndex >= nodeWindowEnd)
    {
        return float3(0.0, 0.0, 1.0);
    }

    float3 tangent = float3(0.0, 0.0, 0.0);
    const float3 currentPosition = EvaluatePathNodePosition(node);
    if (nodeIndex > nodeWindowStart)
    {
        const GpuSuturingPathNode prevNode =
            CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex - 1u);
        if (prevNode.tetIndex != kInvalidSuturingIndex && prevNode.softBodyIndex == node.softBodyIndex)
        {
            tangent += currentPosition - EvaluatePathNodePosition(prevNode);
        }
    }
    if (nodeIndex + 1u < nodeWindowEnd)
    {
        const GpuSuturingPathNode nextNode =
            CRESSIM_SB_LOAD(g_SuturingPathNodes, nodeIndex + 1u);
        if (nextNode.tetIndex != kInvalidSuturingIndex && nextNode.softBodyIndex == node.softBodyIndex)
        {
            tangent += EvaluatePathNodePosition(nextNode) - currentPosition;
        }
    }

    if (dot(tangent, tangent) > kEpsilon)
    {
        return normalize(tangent);
    }

    tangent = node.tangentArcLength.xyz;
    if (dot(tangent, tangent) > kEpsilon)
    {
        return normalize(tangent);
    }

    return float3(0.0, 0.0, 1.0);
}

float ComputeEmbeddedTetMass(float4 p0Inv, float4 p1Inv, float4 p2Inv, float4 p3Inv,
                             float4 barycentrics)
{
    return p0Inv.w * barycentrics.x * barycentrics.x +
           p1Inv.w * barycentrics.y * barycentrics.y +
           p2Inv.w * barycentrics.z * barycentrics.z +
           p3Inv.w * barycentrics.w * barycentrics.w;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint compactIndex = dispatchThreadID.x;
    if (compactIndex >= suturingParticleCount)
    {
        return;
    }
    const uint4 particleRef = CRESSIM_SB_LOAD(g_SuturingParticleRefs, compactIndex);
    const uint particleIndex = particleRef.x;

    const uint strandRole = particleRef.y & 0xffffu;
    const uint ownerType = particleRef.y >> 16u;
    if (strandRole == kParticleStrandRoleNone || strandRole == kParticleStrandRoleNeedleTip)
    {
        return;
    }

    const GpuSuturingInsertionStateStorage state =
        CRESSIM_SB_LOAD(g_SuturingInsertionStates, particleIndex);
    if (state.state != kSuturingInsertionStateInside || state.pathIndex == kInvalidSuturingIndex ||
        state.pathIndex >= suturingPathHeaderCount ||
        state.nearestNodeIndex == kInvalidSuturingIndex ||
        state.nearestNodeIndex >= suturingPathNodeCount)
    {
        return;
    }

    const GpuSuturingPathHeader pathHeader =
        CRESSIM_SB_LOAD(g_SuturingPathHeaders, state.pathIndex);
    if (pathHeader.nodeCount == 0u || state.nearestNodeIndex < pathHeader.nodeStart ||
        state.nearestNodeIndex >= pathHeader.nodeStart + pathHeader.nodeCount)
    {
        return;
    }

    const GpuSuturingPathNode node0 = CRESSIM_SB_LOAD(g_SuturingPathNodes, state.nearestNodeIndex);
    if (node0.tetIndex == kInvalidSuturingIndex || node0.tetIndex >= softTetCount)
    {
        return;
    }
    const uint nodeWindowEnd = min(pathHeader.nodeStart + pathHeader.nodeCount, suturingPathNodeCount);
    const bool hasSegment = pathHeader.nodeCount > 1u && state.nearestNodeIndex + 1u < nodeWindowEnd;
    GpuSuturingPathNode node1 = node0;
    if (hasSegment)
    {
        node1 = CRESSIM_SB_LOAD(g_SuturingPathNodes, state.nearestNodeIndex + 1u);
    }
    if (hasSegment && (node1.tetIndex == kInvalidSuturingIndex || node1.tetIndex >= softTetCount ||
                       node1.softBodyIndex != node0.softBodyIndex))
    {
        return;
    }

    const GpuSoftTet tet0 = CRESSIM_SB_LOAD(g_SoftTets, node0.tetIndex);
    const float4 particlePosInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float4 p0Inv0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet0.particleIndices.x);
    const float4 p1Inv0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet0.particleIndices.y);
    const float4 p2Inv0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet0.particleIndices.z);
    const float4 p3Inv0 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet0.particleIndices.w);
    GpuSoftTet tet1 = tet0;
    float4 p0Inv1 = p0Inv0;
    float4 p1Inv1 = p1Inv0;
    float4 p2Inv1 = p2Inv0;
    float4 p3Inv1 = p3Inv0;
    if (hasSegment && node1.tetIndex != node0.tetIndex)
    {
        tet1 = CRESSIM_SB_LOAD(g_SoftTets, node1.tetIndex);
        p0Inv1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet1.particleIndices.x);
        p1Inv1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet1.particleIndices.y);
        p2Inv1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet1.particleIndices.z);
        p3Inv1 = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, tet1.particleIndices.w);
    }
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
        const float4 rigidPositionInvMass =
            CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex);
        rigidOrientation =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
        invMassRigid = rigidPositionInvMass.w > kEpsilon ? rigidPositionInvMass.w : 0.0;
        invInertiaRigid = invMassRigid > kEpsilon
                              ? CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndex).xyz
                              : 0.0;
        const float3 localProxy = CRESSIM_SB_LOAD(g_RigidProxyLocalPositions, particleIndex).xyz;
        rRigid = QuaternionRotate(rigidOrientation, localProxy);
    }

    const float w0 = p0Inv0.w;
    const float w1 = p1Inv0.w;
    const float w2 = p2Inv0.w;
    const float w3 = p3Inv0.w;

    const float3 pathPosition0 = EvaluatePathNodePosition(node0);
    const float3 pathPosition1 = hasSegment ? EvaluatePathNodePosition(node1) : pathPosition0;
    const float segmentT = hasSegment ? clamp(asfloat(state.closestSegmentTBits), 0.0, 1.0) : 0.0;
    const float3 pathPosition = lerp(pathPosition0, pathPosition1, segmentT);
    float3 tangent = hasSegment ? (pathPosition1 - pathPosition0)
                                : EvaluatePathTangent(state.pathIndex, state.nearestNodeIndex);
    if (dot(tangent, tangent) > kEpsilon)
    {
        tangent = normalize(tangent);
    }
    else
    {
        tangent = EvaluatePathTangent(state.pathIndex, state.nearestNodeIndex);
    }
    const float3 delta = particlePosInvMass.xyz - pathPosition;
    const float3 radial = delta - tangent * dot(delta, tangent);
    const float radialLengthSq = dot(radial, radial);
    const float segmentWeight0 = hasSegment ? (1.0 - segmentT) : 1.0;
    const float segmentWeight1 = hasSegment ? segmentT : 0.0;
    const float effectiveTetMass0 =
        ComputeEmbeddedTetMass(p0Inv0, p1Inv0, p2Inv0, p3Inv0, node0.barycentrics);
    const float effectiveTetMass1 =
        hasSegment ? ComputeEmbeddedTetMass(p0Inv1, p1Inv1, p2Inv1, p3Inv1, node1.barycentrics) : 0.0;
    const float effectiveTetMass = segmentWeight0 * segmentWeight0 * effectiveTetMass0 +
                                   segmentWeight1 * segmentWeight1 * effectiveTetMass1;
    float3 correction = float3(0.0, 0.0, 0.0);

    if (radialLengthSq > 1.0e-8)
    {
        const float radialLength = sqrt(radialLengthSq);
        const float3 radialDirection = radial / radialLength;
        float effectiveRadialParticleMass = effectiveParticleMass;
        if (proxy)
        {
            effectiveRadialParticleMass =
                ComputeContactEffectiveMass(invMassRigid, invInertiaRigid, rigidOrientation, rRigid,
                                            radialDirection);
        }

        const float radialDenom = effectiveRadialParticleMass + effectiveTetMass;
        if (radialDenom > 1.0e-8)
        {
            const float radialLambda = (radialLength / radialDenom) * kSuturingRelaxation;
            float3 radialCorrection = radialDirection * radialLambda;
            const float radialCorrectionLength = length(radialCorrection);
            if (radialCorrectionLength > kSuturingMaxCorrection)
            {
                radialCorrection *=
                    kSuturingMaxCorrection / max(radialCorrectionLength, kEpsilon);
            }
            correction += radialCorrection;
        }
    }

    const float correctionLength = length(correction);
    if (correctionLength <= 1.0e-8)
    {
        return;
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
                                         cross(rRigid, correction)));
    }

    if (p0Inv0.w > kEpsilon && segmentWeight0 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_ParticlePositionCorrections, tet0.particleIndices.x,
            correction * segmentWeight0 * p0Inv0.w * node0.barycentrics.x);
    }
    if (p1Inv0.w > kEpsilon && segmentWeight0 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_ParticlePositionCorrections, tet0.particleIndices.y,
            correction * segmentWeight0 * p1Inv0.w * node0.barycentrics.y);
    }
    if (p2Inv0.w > kEpsilon && segmentWeight0 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_ParticlePositionCorrections, tet0.particleIndices.z,
            correction * segmentWeight0 * p2Inv0.w * node0.barycentrics.z);
    }
    if (p3Inv0.w > kEpsilon && segmentWeight0 > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_ParticlePositionCorrections, tet0.particleIndices.w,
            correction * segmentWeight0 * p3Inv0.w * node0.barycentrics.w);
    }

    if (hasSegment)
    {
        if (p0Inv1.w > kEpsilon && segmentWeight1 > kEpsilon)
        {
            CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
                g_ParticlePositionCorrections, tet1.particleIndices.x,
                correction * segmentWeight1 * p0Inv1.w * node1.barycentrics.x);
        }
        if (p1Inv1.w > kEpsilon && segmentWeight1 > kEpsilon)
        {
            CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
                g_ParticlePositionCorrections, tet1.particleIndices.y,
                correction * segmentWeight1 * p1Inv1.w * node1.barycentrics.y);
        }
        if (p2Inv1.w > kEpsilon && segmentWeight1 > kEpsilon)
        {
            CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
                g_ParticlePositionCorrections, tet1.particleIndices.z,
                correction * segmentWeight1 * p2Inv1.w * node1.barycentrics.z);
        }
        if (p3Inv1.w > kEpsilon && segmentWeight1 > kEpsilon)
        {
            CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
                g_ParticlePositionCorrections, tet1.particleIndices.w,
                correction * segmentWeight1 * p3Inv1.w * node1.barycentrics.w);
        }
    }
}
