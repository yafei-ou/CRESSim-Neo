#include "physics/physics_rigid_dispatch_constants.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "physics/rigid/physics_rigid_solver_shared.hlsli"
#include "physics/rigid/physics_rigid_box_box_manifold.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuColliderContactData, g_ColliderContactData);
CRESSIM_STRUCTURED_BUFFER(GpuCandidatePair, g_CandidatePairs);
CRESSIM_STRUCTURED_BUFFER(GpuNarrowPhaseChunk, g_NarrowPhaseChunks);
CRESSIM_STRUCTURED_BUFFER(GpuNarrowPhaseMeta, g_NarrowPhaseMeta);

CRESSIM_RW_STRUCTURED_BUFFER(uint, g_NarrowPhaseChunkCounter);

CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidContact, g_RigidContacts);

void ClearPairContacts(uint pairIndex)
{
    const uint contactBaseIndex = pairIndex * kRigidContactsPerPair;
    [unroll] for (uint contactOffset = 0u; contactOffset < kRigidContactsPerPair; ++contactOffset)
    {
        GpuRigidContact cleared;
        cleared.bodyA = 0u;
        cleared.bodyB = 0u;
        cleared.active = 0u;
        cleared.reserved = 0u;
        cleared.normalPenetration = 0.0;
        cleared.localPointA = 0.0;
        cleared.localPointB = 0.0;
        cleared.material = 0.0;
        CRESSIM_SB_STORE(g_RigidContacts, contactBaseIndex + contactOffset, cleared);
    }
}

void ProcessPair(uint pairIndex, uint pairType)
{
    ClearPairContacts(pairIndex);
    const uint contactBaseIndex = pairIndex * kRigidContactsPerPair;

    const GpuCandidatePair pair = CRESSIM_SB_LOAD(g_CandidatePairs, pairIndex);
    const uint colliderA = pair.colliderA;
    const uint colliderB = pair.colliderB;
    const GpuColliderContactData colliderDataA = CRESSIM_SB_LOAD(g_ColliderContactData, colliderA);
    const GpuColliderContactData colliderDataB = CRESSIM_SB_LOAD(g_ColliderContactData, colliderB);
    const uint bodyA = colliderDataA.ownerBody;
    const uint bodyB = colliderDataB.ownerBody;
    if (bodyA >= rigidBodyCount || bodyB >= rigidBodyCount || bodyA == bodyB)
    {
        return;
    }

    const float4 bodyPositionInvMassA = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyA);
    const float4 bodyPositionInvMassB = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyB);
    const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyA);
    const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyB);
    if (bodyTypeA != kRigidBodyTypeDynamic && bodyTypeB != kRigidBodyTypeDynamic)
    {
        return;
    }

    const float4 bodyOrientationA =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyA));
    const float4 bodyOrientationB =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyB));
    const float4 colliderParamsA = colliderDataA.shapeParams;
    const float4 colliderParamsB = colliderDataB.shapeParams;
    const float4 scaleA = CRESSIM_SB_LOAD(g_RigidBodyScales, bodyA);
    const float4 scaleB = CRESSIM_SB_LOAD(g_RigidBodyScales, bodyB);
    const uint shapeTypeA = colliderDataA.shapeType;
    const uint shapeTypeB = colliderDataB.shapeType;
    const float3 contactMaterial = CombineContactMaterial(colliderDataA.material,
                                                          colliderDataB.material);
    const float3 colliderPositionA = ComposeColliderWorldPosition(
        bodyPositionInvMassA.xyz, bodyOrientationA,
        colliderDataA.localPosition.xyz * scaleA.xyz);
    const float3 colliderPositionB = ComposeColliderWorldPosition(
        bodyPositionInvMassB.xyz, bodyOrientationB,
        colliderDataB.localPosition.xyz * scaleB.xyz);
    const float4 colliderOrientationA = ComposeColliderWorldOrientation(
        bodyOrientationA, QuaternionNormalize(colliderDataA.localOrientation));
    const float4 colliderOrientationB = ComposeColliderWorldOrientation(
        bodyOrientationB, QuaternionNormalize(colliderDataB.localOrientation));

    float3 aabbMinA;
    float3 aabbMaxA;
    float3 aabbMinB;
    float3 aabbMaxB;
    ComputeBodyAabb(shapeTypeA, colliderPositionA, colliderOrientationA, colliderParamsA, scaleA,
                    aabbMinA, aabbMaxA);
    ComputeBodyAabb(shapeTypeB, colliderPositionB, colliderOrientationB, colliderParamsB, scaleB,
                    aabbMinB, aabbMaxB);
    if (!AabbOverlaps(aabbMinA, aabbMaxA, aabbMinB, aabbMaxB))
    {
        return;
    }

    float3 normalAtoB = 0.0;
    float3 pointAWorld = 0.0;
    float3 pointBWorld = 0.0;
    float penetration = 0.0;

    if (pairType == 3u)
    {
        float3 satNormal = 0.0;
        float satPenetration = 0.0;
        uint satAxisType = kObbAxisFaceA;
        uint satAxisA = 0u;
        uint satAxisB = 0u;
        uint manifoldContactSource = 0u;
        if (!ComputeBoxBoxSat(colliderPositionA, colliderOrientationA,
                              BoxHalfExtents(colliderParamsA, scaleA), colliderPositionB,
                              colliderOrientationB, BoxHalfExtents(colliderParamsB, scaleB),
                              satNormal, satPenetration, satAxisType, satAxisA, satAxisB))
        {
            return;
        }

        const float3 contactNormal = SafeNormalize(satNormal, float3(0.0, 1.0, 0.0));

        GpuRigidContact manifoldContacts[kRigidContactsPerPair];
        [unroll] for (uint i = 0u; i < kRigidContactsPerPair; ++i)
        {
            GpuRigidContact cleared;
            cleared.bodyA = bodyA;
            cleared.bodyB = bodyB;
            cleared.active = 0u;
            cleared.reserved = 0u;
            cleared.normalPenetration = 0.0;
            cleared.localPointA = 0.0;
            cleared.localPointB = 0.0;
            cleared.material = 0.0;
            manifoldContacts[i] = cleared;
        }

        const uint manifoldCount =
            GenerateBoxBoxManifoldContacts(bodyA, bodyB, bodyPositionInvMassA.xyz,
                                           bodyOrientationA, colliderPositionA,
                                           colliderOrientationA, colliderParamsA, scaleA,
                                           bodyPositionInvMassB.xyz, bodyOrientationB,
                                           colliderPositionB, colliderOrientationB,
                                           colliderParamsB, scaleB, contactNormal,
                                           contactMaterial,
                                           manifoldContactSource,
                                           manifoldContacts);
        if (manifoldCount > 0u)
        {
            [unroll] for (uint i = 0u; i < kRigidContactsPerPair; ++i)
            {
                if (i >= manifoldCount)
                {
                    break;
                }
                CRESSIM_SB_STORE(g_RigidContacts, contactBaseIndex + i, manifoldContacts[i]);
            }
            return;
        }
    }

    if (!GenerateRigidContact(shapeTypeA, colliderPositionA, colliderOrientationA, colliderParamsA,
                              scaleA, shapeTypeB, colliderPositionB, colliderOrientationB,
                              colliderParamsB, scaleB,
                              normalAtoB, pointAWorld, pointBWorld, penetration) ||
        penetration <= 0.0)
    {
        return;
    }

    const float3 contactNormal = SafeNormalize(normalAtoB, float3(0.0, 1.0, 0.0));

    GpuRigidContact contact;
    contact.bodyA = bodyA;
    contact.bodyB = bodyB;
    contact.active = 1u;
    contact.reserved = (pairType == 3u)
                           ? EncodeBoxBoxContactReserved(kBoxBoxContactSourceGenericFallback)
                           : 0u;
    contact.normalPenetration = float4(contactNormal, penetration);
    contact.material = float4(contactMaterial, 0.0);
    contact.localPointA =
        float4(QuaternionInverseRotate(bodyOrientationA, pointAWorld - bodyPositionInvMassA.xyz),
               1.0);
    contact.localPointB =
        float4(QuaternionInverseRotate(bodyOrientationB, pointBWorld - bodyPositionInvMassB.xyz),
               1.0);
    CRESSIM_SB_STORE(g_RigidContacts, contactBaseIndex, contact);
}

groupshared uint s_ChunkId;
groupshared uint s_ChunkPairType;
groupshared uint s_ChunkPairStart;
groupshared uint s_ChunkPairCount;

[numthreads(128, 1, 1)] void main(uint3 groupThreadID : SV_GroupThreadID)
{
    const uint totalChunks = CRESSIM_SB_LOAD(g_NarrowPhaseMeta, 0).chunkCount;
    if (totalChunks == 0u)
    {
        return;
    }

    while (true)
    {
        if (groupThreadID.x == 0u)
        {
            InterlockedAdd(CRESSIM_SB_REF(g_NarrowPhaseChunkCounter, 0u), 1u, s_ChunkId);
            if (s_ChunkId < totalChunks)
            {
                const GpuNarrowPhaseChunk chunk = CRESSIM_SB_LOAD(g_NarrowPhaseChunks, s_ChunkId);
                s_ChunkPairType = chunk.pairType;
                s_ChunkPairStart = chunk.pairStart;
                s_ChunkPairCount = chunk.pairCount;
            }
            else
            {
                s_ChunkPairType = 0u;
                s_ChunkPairStart = 0u;
                s_ChunkPairCount = 0u;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (s_ChunkId >= totalChunks)
        {
            return;
        }

        const uint localPairOffset = groupThreadID.x;
        if (localPairOffset < s_ChunkPairCount)
        {
            ProcessPair(s_ChunkPairStart + localPairOffset, s_ChunkPairType);
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
