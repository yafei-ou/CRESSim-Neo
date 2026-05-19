#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"
#include "../../../include/physics/collision/physics_shape_common.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint4, g_ParticleBroadPhaseMetadata);

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);

CRESSIM_STRUCTURED_BUFFER(uint2, g_BodyColliderRanges);
CRESSIM_STRUCTURED_BUFFER(uint, g_BodyColliderIndices);

CRESSIM_STRUCTURED_BUFFER(GpuColliderGeometryData, g_ColliderGeometryData);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);

CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_ParticleCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleRigidContact, g_ParticleRigidContacts);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (pairIndex >= meta.particleRigidCandidateCount)
    {
        return;
    }

    GpuParticleRigidContact outContact = (GpuParticleRigidContact)0;

    const GpuParticleCandidatePair pair = CRESSIM_SB_LOAD(g_ParticleCandidatePairs, pairIndex);

    const uint particleIndex = pair.indexA;
    const uint rigidBodyIndex = pair.indexB;
    const float3 softPosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const float softRadius = CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex);
    const uint4 softMetadata = CRESSIM_SB_LOAD(g_ParticleBroadPhaseMetadata, particleIndex);
    const uint softEnvironment = softMetadata.x;
    const uint softLayer = softMetadata.z;
    const uint softMask = softMetadata.w;

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
    float3 bestContactWorld = float3(0.0, 0.0, 0.0);

    [loop]
    for (uint i = 0u; i < kParticleRigidColliderIterationCap; ++i)
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

        const GpuColliderGeometryData colliderGeometry =
            CRESSIM_SB_LOAD(g_ColliderGeometryData, colliderIndex);
        const float3 colliderLocalPosition = colliderGeometry.localPosition.xyz;
        const float4 colliderLocalOrientation =
            QuaternionNormalize(colliderGeometry.localOrientation);
        const float3 colliderWorldPosition =
            ComposeColliderWorldPosition(bodyPosition, bodyOrientation, colliderLocalPosition);
        const float4 colliderWorldOrientation =
            ComposeColliderWorldOrientation(bodyOrientation, colliderLocalOrientation);
        const float3 particleLocal =
            QuaternionInverseRotate(colliderWorldOrientation, softPosition - colliderWorldPosition);

        float3 normalLocal = float3(0.0, 1.0, 0.0);
        float3 candidateContactWorld = float3(0.0, 0.0, 0.0);
        float signedDistance = 0.0;
        const uint shapeType = broadPhase.shapeType;
        const float4 shapeParams = colliderGeometry.shapeParams;

        if (shapeType == kColliderSphere)
        {
            const float radius = SphereRadius(shapeParams, bodyScale);
            const float length = sqrt(dot(particleLocal, particleLocal));
            normalLocal = length > kEpsilon ? (particleLocal / length) : float3(0.0, 1.0, 0.0);
            signedDistance = length - radius;
            candidateContactWorld =
                colliderWorldPosition + QuaternionRotate(colliderWorldOrientation, normalLocal * radius);
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
                candidateContactWorld =
                    colliderWorldPosition + QuaternionRotate(colliderWorldOrientation, closest);
            }
            else
            {
                normalLocal = ClosestFaceNormalLocal(particleLocal, halfExtents);
                const float3 distances = halfExtents - abs(particleLocal);
                signedDistance = -min(distances.x, min(distances.y, distances.z));
                float3 closestOnSurface = particleLocal;
                if (abs(normalLocal.x) > 0.5)
                {
                    closestOnSurface.x = normalLocal.x * halfExtents.x;
                }
                else if (abs(normalLocal.y) > 0.5)
                {
                    closestOnSurface.y = normalLocal.y * halfExtents.y;
                }
                else
                {
                    closestOnSurface.z = normalLocal.z * halfExtents.z;
                }
                candidateContactWorld = colliderWorldPosition +
                                        QuaternionRotate(colliderWorldOrientation, closestOnSurface);
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
            candidateContactWorld = closest + worldNormal * capsuleRadius;
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
        bestContactWorld = candidateContactWorld;
    }

    if (bestPenetration > 0.0)
    {
        outContact.particleIndex = particleIndex;
        outContact.rigidBodyIndex = rigidBodyIndex;
        outContact.colliderIndex = bestColliderIndex;
        outContact.active = 1u;
        outContact.normalPenetration = float4(bestNormal, bestPenetration);
        outContact.rigidLocalPoint =
            float4(QuaternionInverseRotate(bodyOrientation, bestContactWorld - bodyPosition), 0.0);
    }

    CRESSIM_SB_STORE(g_ParticleRigidContacts, pairIndex, outContact);
    CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, outContact.active);
}
