#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);
CRESSIM_RW_STRUCTURED_BUFFER(GpuProxyRigidContactMeta, g_ProxyRigidContactMeta);

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyProxyParticleContactMaterials);

CRESSIM_STRUCTURED_BUFFER(uint2, g_BodyColliderRanges);
CRESSIM_STRUCTURED_BUFFER(uint, g_BodyColliderIndices);
CRESSIM_STRUCTURED_BUFFER(GpuColliderContactData, g_ColliderContactData);

CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_ParticleCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidContact, g_RigidContacts);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta neighborMeta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (pairIndex >= neighborMeta.particleRigidCandidateCount)
    {
        return;
    }

    const GpuParticleCandidatePair pair = CRESSIM_SB_LOAD(g_ParticleCandidatePairs, pairIndex);
    const uint particleIndex = pair.indexA;
    const uint ownerType = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleIndex);
    if (ownerType != kParticleOwnerTypeRigidBody)
    {
        return;
    }

    const uint bodyA = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleIndex);
    const uint bodyB = pair.indexB;
    if (bodyA == bodyB)
    {
        return;
    }

    const float4 bodyAPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyA);
    const float4 bodyBPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyB);
    if (bodyAPositionInvMass.w <= kEpsilon && bodyBPositionInvMass.w <= kEpsilon)
    {
        return;
    }

    const float3 sphereCenter = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const float sphereRadius = CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex);
    const float4 bodyAOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyA));
    const float4 bodyBOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyB));
    const float4 bodyBScale = CRESSIM_SB_LOAD(g_RigidBodyScales, bodyB);
    const float4 proxyMaterial =
        CRESSIM_SB_LOAD(g_RigidBodyProxyParticleContactMaterials, bodyA);

    const uint2 colliderRange = CRESSIM_SB_LOAD(g_BodyColliderRanges, bodyB);
    const uint colliderOffset = colliderRange.x;
    const uint colliderCount = colliderRange.y;

    float bestPenetration = 0.0;
    float3 bestNormal = float3(0.0, 1.0, 0.0);
    float3 bestPointA = 0.0;
    float3 bestPointB = 0.0;
    float3 bestMaterial = 0.0;

    [loop]
    for (uint i = 0u; i < kParticleRigidColliderIterationCap; ++i)
    {
        if (i >= colliderCount)
        {
            break;
        }

        const uint colliderIndex = CRESSIM_SB_LOAD(g_BodyColliderIndices, colliderOffset + i);
        const GpuColliderContactData collider =
            CRESSIM_SB_LOAD(g_ColliderContactData, colliderIndex);
        if (collider.reserved0 == 0u)
        {
            continue;
        }

        const float3 colliderWorldPosition = ComposeColliderWorldPosition(
            bodyBPositionInvMass.xyz, bodyBOrientation, collider.localPosition.xyz);
        const float4 colliderWorldOrientation = ComposeColliderWorldOrientation(
            bodyBOrientation, QuaternionNormalize(collider.localOrientation));

        float3 normal = 0.0;
        float3 pointA = 0.0;
        float3 pointB = 0.0;
        float penetration = 0.0;
        bool hit = false;

        if (collider.shapeType == kColliderSphere)
        {
            hit = GenerateSphereSphereContactPoints(
                sphereCenter, sphereRadius, colliderWorldPosition,
                SphereRadius(collider.shapeParams, bodyBScale), normal, pointA, pointB, penetration);
        }
        else if (collider.shapeType == kColliderBox)
        {
            hit = GenerateSphereBoxContactPoints(
                sphereCenter, sphereRadius, colliderWorldPosition, colliderWorldOrientation,
                BoxHalfExtents(collider.shapeParams, bodyBScale), normal, pointA, pointB, penetration);
        }
        else
        {
            float3 capsuleA, capsuleB;
            float capsuleRadius = 0.0;
            CapsuleSegment(colliderWorldPosition, colliderWorldOrientation, collider.shapeParams,
                           bodyBScale, capsuleA, capsuleB, capsuleRadius);
            hit = GenerateSphereCapsuleContactPoints(
                sphereCenter, sphereRadius, capsuleA, capsuleB, capsuleRadius, normal, pointA,
                pointB, penetration);
        }

        if (!hit || penetration <= bestPenetration)
        {
            continue;
        }

        bestPenetration = penetration;
        bestNormal = SafeNormalize(normal, float3(0.0, 1.0, 0.0));
        bestPointA = pointA;
        bestPointB = pointB;
        bestMaterial = CombineContactMaterial(proxyMaterial, collider.material);
    }

    if (bestPenetration <= 0.0)
    {
        return;
    }

    uint requiredContactIndex = 0u;
    InterlockedAdd(CRESSIM_SB_REF(g_ProxyRigidContactMeta, 0u).requiredContactCount, 1u,
                   requiredContactIndex);

    const uint primitiveContactCount =
        CRESSIM_SB_LOAD(g_BroadPhaseMeta, 0u).candidatePairCount * kRigidContactsPerPair;
    const uint proxyContactCapacity =
        contactCapacity > primitiveContactCount ? (contactCapacity - primitiveContactCount) : 0u;
    if (requiredContactIndex >= proxyContactCapacity)
    {
        uint ignored = 0u;
        InterlockedAdd(CRESSIM_SB_REF(g_ProxyRigidContactMeta, 0u).overflow, 1u, ignored);
        return;
    }

    uint proxyContactIndex = 0u;
    InterlockedAdd(CRESSIM_SB_REF(g_ProxyRigidContactMeta, 0u).activeContactCount, 1u,
                   proxyContactIndex);
    const uint contactIndex = primitiveContactCount + proxyContactIndex;

    GpuRigidContact contact;
    contact.bodyA = bodyA;
    contact.bodyB = bodyB;
    contact.active = 1u;
    contact.reserved = 0u;
    contact.normalPenetration = float4(bestNormal, bestPenetration);
    contact.localPointA =
        float4(QuaternionInverseRotate(bodyAOrientation, bestPointA - bodyAPositionInvMass.xyz), 0.0);
    contact.localPointB =
        float4(QuaternionInverseRotate(bodyBOrientation, bestPointB - bodyBPositionInvMass.xyz), 0.0);
    contact.material = float4(bestMaterial, 0.0);
    CRESSIM_SB_STORE(g_RigidContacts, contactIndex, contact);
}
