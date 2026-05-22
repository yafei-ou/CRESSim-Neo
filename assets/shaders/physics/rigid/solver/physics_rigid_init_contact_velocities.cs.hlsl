#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(GpuRigidContact, g_RigidContacts);
CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);

CRESSIM_GLOBALLYCOHERENT_RW_STRUCTURED_BUFFER(GpuRigidBodyPairContactAggregateHeader,
                                              g_RigidBodyPairAggregateHeaders);
CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidBodyPairContactAggregateSlot,
                             g_RigidBodyPairAggregateSlots);

uint HashBodyPair(uint bodyA, uint bodyB)
{
    return (bodyA * 73856093u) ^ (bodyB * 19349663u);
}

GpuRigidContact CanonicalizeContact(GpuRigidContact contact)
{
    if (contact.bodyA <= contact.bodyB)
    {
        return contact;
    }

    GpuRigidContact swapped = contact;
    swapped.bodyA = contact.bodyB;
    swapped.bodyB = contact.bodyA;
    swapped.normalPenetration = float4(-contact.normalPenetration.xyz, contact.normalPenetration.w);
    swapped.localPointA = contact.localPointB;
    swapped.localPointB = contact.localPointA;
    return swapped;
}

float ComputeContactNormalVelocity(GpuRigidContact contact)
{
    const uint bodyA = contact.bodyA;
    const uint bodyB = contact.bodyB;
    const float4 positionInvMassA =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyA);
    const float4 positionInvMassB =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyB);
    const float4 orientationA =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyA));
    const float4 orientationB =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyB));
    const float3 pointA =
        positionInvMassA.xyz + QuaternionRotate(orientationA, contact.localPointA.xyz);
    const float3 pointB =
        positionInvMassB.xyz + QuaternionRotate(orientationB, contact.localPointB.xyz);
    const float3 rA = pointA - positionInvMassA.xyz;
    const float3 rB = pointB - positionInvMassB.xyz;
    const float3 velocityA = ComputeContactPointVelocity(
        CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, bodyA).xyz,
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, bodyA).xyz, rA);
    const float3 velocityB = ComputeContactPointVelocity(
        CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, bodyB).xyz,
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, bodyB).xyz, rB);
    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    return dot(velocityB - velocityA, normal);
}

uint ReserveAggregateEntry(uint bodyA, uint bodyB)
{
    if (candidatePairCapacity == 0u)
    {
        return 0xffffffffu;
    }

    const uint startIndex = HashBodyPair(bodyA, bodyB) % candidatePairCapacity;
    [loop] for (uint probe = 0u; probe < candidatePairCapacity; ++probe)
    {
        const uint aggregateIndex = (startIndex + probe) % candidatePairCapacity;

        // Bound the wait on in-flight entry initialization so a bad header state
        // cannot trap the whole dispatch in an infinite UAV spin loop.
        [loop] for (uint spin = 0u; spin < 64u; ++spin)
        {
            const GpuRigidBodyPairContactAggregateHeader header =
                CRESSIM_SB_LOAD(g_RigidBodyPairAggregateHeaders, aggregateIndex);

            if (header.flags == 0u)
            {
                uint previousFlags = 0u;
                InterlockedCompareExchange(
                    CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).flags,
                    0u, kRigidAggregateEntryFlagInitializing, previousFlags);
                if (previousFlags == 0u)
                {
                    CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).bodyA = bodyA;
                    CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).bodyB = bodyB;
                    CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).count = 0u;
                    DeviceMemoryBarrier();
                    CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).flags =
                        kRigidAggregateEntryFlagReady;
                    return aggregateIndex;
                }
                continue;
            }

            if ((header.flags & kRigidAggregateEntryFlagReady) != 0u)
            {
                if (header.bodyA == bodyA && header.bodyB == bodyB)
                {
                    return aggregateIndex;
                }
                break;
            }

            if ((header.flags & kRigidAggregateEntryFlagInitializing) != 0u)
            {
                continue;
            }

            break;
        }
    }

    return 0xffffffffu;
}

bool ReserveAggregateSlot(uint aggregateIndex, out uint slotIndex)
{
    slotIndex = 0xffffffffu;

    [loop] for (uint attempt = 0u; attempt < (kRigidBodyPairAggregateContacts + 1u); ++attempt)
    {
        const uint currentCount =
            CRESSIM_SB_LOAD(g_RigidBodyPairAggregateHeaders, aggregateIndex).count;
        if (currentCount >= kRigidBodyPairAggregateContacts)
        {
            InterlockedOr(CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).flags,
                          kRigidAggregateEntryFlagOverflow);
            return false;
        }

        uint previousCount = 0u;
        InterlockedCompareExchange(
            CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).count,
            currentCount, currentCount + 1u, previousCount);
        if (previousCount == currentCount)
        {
            slotIndex = currentCount;
            return true;
        }
    }

    InterlockedOr(CRESSIM_SB_REF(g_RigidBodyPairAggregateHeaders, aggregateIndex).flags,
                  kRigidAggregateEntryFlagOverflow);
    return false;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const GpuBroadPhaseMeta broadPhaseMeta = CRESSIM_SB_LOAD(g_BroadPhaseMeta, 0u);
    const uint totalContactSlots = broadPhaseMeta.candidatePairCount * kRigidContactsPerPair;
    if (contactIndex >= totalContactSlots)
    {
        return;
    }

    const GpuRigidContact rawContact = CRESSIM_SB_LOAD(g_RigidContacts, contactIndex);
    if (rawContact.active == 0u)
    {
        return;
    }

    const GpuRigidContact contact = CanonicalizeContact(rawContact);
    const uint aggregateIndex = ReserveAggregateEntry(contact.bodyA, contact.bodyB);
    if (aggregateIndex == 0xffffffffu)
    {
        return;
    }

    uint slotIndex = 0xffffffffu;
    if (!ReserveAggregateSlot(aggregateIndex, slotIndex))
    {
        return;
    }

    GpuRigidBodyPairContactAggregateSlot aggregateSlot;
    aggregateSlot.normal =
        float4(SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0)), 0.0);
    aggregateSlot.localPointA = contact.localPointA;
    aggregateSlot.localPointB = contact.localPointB;
    aggregateSlot.solverState = 0.0;

    const float normalVelocity = ComputeContactNormalVelocity(contact);
    if (normalVelocity < -kRigidRestitutionThreshold)
    {
        const float restitution = saturate(contact.material.y);
        aggregateSlot.solverState.y = -restitution * normalVelocity;
    }

    const uint flatSlotIndex = aggregateIndex * kRigidBodyPairAggregateContacts + slotIndex;
    CRESSIM_SB_STORE(g_RigidBodyPairAggregateSlots, flatSlotIndex, aggregateSlot);
}
