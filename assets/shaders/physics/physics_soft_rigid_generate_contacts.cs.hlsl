#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_SoftParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint4, g_SoftParticleBroadPhaseMetadata);

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);

CRESSIM_STRUCTURED_BUFFER(uint2, g_BodyColliderRanges);
CRESSIM_STRUCTURED_BUFFER(uint, g_BodyColliderIndices);

CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderShapeParams);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderLocalPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderLocalOrientations);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);

CRESSIM_STRUCTURED_BUFFER(GpuSoftRigidCandidatePair, g_SoftRigidCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(uint, g_SoftRigidCandidatePairCount);

CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftRigidContact, g_SoftRigidContacts);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    if (pairIndex >= softRigidCandidatePairCapacity)
    {
        return;
    }

    GpuSoftRigidContact outContact;
    outContact.softParticleIndex = 0u;
    outContact.rigidBodyIndex = 0u;
    outContact.colliderIndex = 0u;
    outContact.active = 0u;
    outContact.normalPenetration = float4(0.0, 0.0, 0.0, 0.0);

    const uint validPairCount = CRESSIM_SB_LOAD(g_SoftRigidCandidatePairCount, 0u);
    if (pairIndex >= validPairCount)
    {
        CRESSIM_SB_STORE(g_SoftRigidContacts, pairIndex, outContact);
        return;
    }

    const GpuSoftRigidCandidatePair pair = CRESSIM_SB_LOAD(g_SoftRigidCandidatePairs, pairIndex);
    if (pair.pairType != kSoftRigidCandidatePairTypeSoftRigid)
    {
        CRESSIM_SB_STORE(g_SoftRigidContacts, pairIndex, outContact);
        return;
    }

    const uint softParticleIndex = pair.indexA;
    const uint rigidBodyIndex = pair.indexB;
    const float3 softPosition =
        CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, softParticleIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_SoftParticleRadii, softParticleIndex);
    const uint4 softMetadata = CRESSIM_SB_LOAD(g_SoftParticleBroadPhaseMetadata, softParticleIndex);
    const uint softEnvironment = softMetadata.x;
    const uint softLayer = softMetadata.y;
    const uint softMask = softMetadata.z;

    const float3 bodyPosition =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex).xyz;
    const float4 bodyOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
    const float4 bodyScale = CRESSIM_SB_LOAD(g_RigidBodyScales, rigidBodyIndex);

    const uint2 colliderRange = CRESSIM_SB_LOAD(g_BodyColliderRanges, rigidBodyIndex);
    const uint colliderOffset = colliderRange.x;
    const uint colliderCount = colliderRange.y;

    float bestPenetration = 0.0;
    uint bestColliderIndex = 0u;
    float3 bestNormal = float3(0.0, 1.0, 0.0);

    [loop]
    for (uint i = 0u; i < 64u; ++i)
    {
        if (i >= colliderCount)
        {
            break;
        }

        const uint colliderIndex = CRESSIM_SB_LOAD(g_BodyColliderIndices, colliderOffset + i);
        const GpuColliderBroadPhaseData broadPhase =
            CRESSIM_SB_LOAD(g_ColliderBroadPhaseData, colliderIndex);
        if (broadPhase.enabledFlag == 0u)
        {
            continue;
        }
        if (broadPhase.environmentIndex != softEnvironment)
        {
            continue;
        }
        if ((softMask & broadPhase.collisionLayer) == 0u ||
            (broadPhase.collisionMask & softLayer) == 0u)
        {
            continue;
        }

        const float3 colliderLocalPosition =
            CRESSIM_SB_LOAD(g_ColliderLocalPositions, colliderIndex).xyz;
        const float4 colliderLocalOrientation =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_ColliderLocalOrientations, colliderIndex));
        const float3 colliderWorldPosition =
            ComposeColliderWorldPosition(bodyPosition, bodyOrientation, colliderLocalPosition);
        const float4 colliderWorldOrientation =
            ComposeColliderWorldOrientation(bodyOrientation, colliderLocalOrientation);
        const float3 particleLocal =
            QuaternionInverseRotate(colliderWorldOrientation, softPosition - colliderWorldPosition);

        float3 normalLocal = float3(0.0, 1.0, 0.0);
        float signedDistance = 0.0;
        const uint shapeType = broadPhase.shapeType;
        const float4 shapeParams = CRESSIM_SB_LOAD(g_ColliderShapeParams, colliderIndex);

        if (shapeType == kColliderSphere)
        {
            const float radius = SphereRadius(shapeParams, bodyScale);
            const float length = sqrt(dot(particleLocal, particleLocal));
            normalLocal = length > kEpsilon ? (particleLocal / length) : float3(0.0, 1.0, 0.0);
            signedDistance = length - radius;
        }
        else if (shapeType == kColliderBox)
        {
            const float3 halfExtents = BoxHalfExtents(shapeParams, bodyScale);
            const float3 closest = clamp(particleLocal, -halfExtents, halfExtents);
            const float3 delta = particleLocal - closest;
            const float deltaLength = sqrt(dot(delta, delta));
            if (deltaLength > kEpsilon)
            {
                normalLocal = delta / deltaLength;
                signedDistance = deltaLength;
            }
            else
            {
                normalLocal = ClosestFaceNormalLocal(particleLocal, halfExtents);
                const float3 distances = halfExtents - abs(particleLocal);
                signedDistance = -min(distances.x, min(distances.y, distances.z));
            }
        }
        else
        {
            float3 segA, segB;
            float capsuleRadius = 0.0;
            CapsuleSegment(colliderWorldPosition, colliderWorldOrientation, shapeParams, bodyScale,
                           segA, segB, capsuleRadius);
            const float3 seg = segB - segA;
            const float segLengthSq = dot(seg, seg);
            float t = 0.0;
            if (segLengthSq > kEpsilon)
            {
                t = saturate(dot(softPosition - segA, seg) / segLengthSq);
            }
            const float3 closest = lerp(segA, segB, t);
            const float3 delta = softPosition - closest;
            const float length = sqrt(dot(delta, delta));
            const float3 worldNormal =
                length > kEpsilon ? (delta / length) : float3(1.0, 0.0, 0.0);
            signedDistance = length - capsuleRadius;
            normalLocal = QuaternionInverseRotate(colliderWorldOrientation, worldNormal);
        }

        const float penetration = softRadius - signedDistance;
        if (penetration <= 0.0 || penetration <= bestPenetration)
        {
            continue;
        }

        bestPenetration = penetration;
        bestColliderIndex = colliderIndex;
        bestNormal = SafeNormalize(QuaternionRotate(colliderWorldOrientation, normalLocal),
                                   float3(0.0, 1.0, 0.0));
    }

    if (bestPenetration > 0.0)
    {
        outContact.softParticleIndex = softParticleIndex;
        outContact.rigidBodyIndex = rigidBodyIndex;
        outContact.colliderIndex = bestColliderIndex;
        outContact.active = 1u;
        outContact.normalPenetration = float4(bestNormal, bestPenetration);
    }

    CRESSIM_SB_STORE(g_SoftRigidContacts, pairIndex, outContact);
}
