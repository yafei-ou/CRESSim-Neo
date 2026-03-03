cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint pairCount;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
    uint reserved0;
    uint reserved1;
};

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyScales;
StructuredBuffer<uint> g_RigidBodyColliderShapeTypes;
StructuredBuffer<float4> g_RigidBodyColliderParams;

RWStructuredBuffer<GpuRigidContact> g_RigidContacts;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    if (pairIndex >= pairCount)
    {
        return;
    }

    const uint contactBaseIndex = pairIndex * kRigidContactsPerPair;
    [unroll]
    for (uint contactOffset = 0u; contactOffset < kRigidContactsPerPair; ++contactOffset)
    {
        GpuRigidContact cleared;
        cleared.bodyA = 0u;
        cleared.bodyB = 0u;
        cleared.active = 0u;
        cleared.reserved = 0u;
        cleared.normalPenetration = 0.0;
        cleared.worldPoint = 0.0;
        g_RigidContacts[contactBaseIndex + contactOffset] = cleared;
    }

    uint bodyA = 0u;
    uint bodyB = 0u;
    PairIndexToBodies(pairIndex, rigidBodyCount, bodyA, bodyB);

    const float4 positionInvMassA = g_PredictedRigidBodyPositionsInvMass[bodyA];
    const float4 positionInvMassB = g_PredictedRigidBodyPositionsInvMass[bodyB];
    if (positionInvMassA.w == 0.0 && positionInvMassB.w == 0.0)
    {
        return;
    }

    const float4 orientationA = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyA]);
    const float4 orientationB = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyB]);
    const uint shapeTypeA = g_RigidBodyColliderShapeTypes[bodyA];
    const uint shapeTypeB = g_RigidBodyColliderShapeTypes[bodyB];
    const float4 colliderParamsA = g_RigidBodyColliderParams[bodyA];
    const float4 colliderParamsB = g_RigidBodyColliderParams[bodyB];
    const float4 scaleA = g_RigidBodyScales[bodyA];
    const float4 scaleB = g_RigidBodyScales[bodyB];

    float3 aabbMinA;
    float3 aabbMaxA;
    float3 aabbMinB;
    float3 aabbMaxB;
    ComputeBodyAabb(shapeTypeA, positionInvMassA.xyz, orientationA, colliderParamsA, scaleA,
                    aabbMinA, aabbMaxA);
    ComputeBodyAabb(shapeTypeB, positionInvMassB.xyz, orientationB, colliderParamsB, scaleB,
                    aabbMinB, aabbMaxB);
    if (!AabbOverlaps(aabbMinA, aabbMaxA, aabbMinB, aabbMaxB))
    {
        return;
    }

    float3 normal = 0.0;
    float3 contactPoint = 0.0;
    float penetration = 0.0;
    if (!GenerateRigidContact(shapeTypeA, positionInvMassA.xyz, orientationA, colliderParamsA,
                              scaleA, shapeTypeB, positionInvMassB.xyz, orientationB,
                              colliderParamsB, scaleB, normal, contactPoint, penetration) ||
        penetration <= 0.0)
    {
        return;
    }

    GpuRigidContact contact;
    contact.bodyA = bodyA;
    contact.bodyB = bodyB;
    contact.active = 1u;
    contact.reserved = 0u;
    contact.normalPenetration = float4(SafeNormalize(normal, float3(0.0, 1.0, 0.0)), penetration);
    contact.worldPoint = float4(contactPoint, 1.0);
    g_RigidContacts[contactBaseIndex] = contact;
}
