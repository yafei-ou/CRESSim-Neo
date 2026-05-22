#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"
#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRigidBodyPairContactAggregateHeader,
                          g_RigidBodyPairAggregateHeaders);

CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidBodyPairContactAggregateSlot,
                             g_RigidBodyPairAggregateSlots);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyAngularVelocityCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint aggregateIndex = dispatchThreadID.x;
    if (aggregateIndex >= candidatePairCapacity)
    {
        return;
    }

    const GpuRigidBodyPairContactAggregateHeader aggregateHeader =
        CRESSIM_SB_LOAD(g_RigidBodyPairAggregateHeaders, aggregateIndex);
    if ((aggregateHeader.flags & kRigidAggregateEntryFlagReady) == 0u ||
        aggregateHeader.count == 0u)
    {
        return;
    }

    const uint bodyA = aggregateHeader.bodyA;
    const uint bodyB = aggregateHeader.bodyB;
    const float4 positionInvMassA = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyA);
    const float4 positionInvMassB = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyB);
    const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyA);
    const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyB);
    const float invMassA = bodyTypeA == kRigidBodyTypeDynamic ? positionInvMassA.w : 0.0;
    const float invMassB = bodyTypeB == kRigidBodyTypeDynamic ? positionInvMassB.w : 0.0;
    if (invMassA == 0.0 && invMassB == 0.0)
    {
        return;
    }

    const float4 orientationA =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyA));
    const float4 orientationB =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyB));
    float3 invInertiaA = CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, bodyA).xyz;
    float3 invInertiaB = CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, bodyB).xyz;
    if (invMassA == 0.0)
        invInertiaA = 0.0;
    if (invMassB == 0.0)
        invInertiaB = 0.0;

    const float3 linearVelocityA0 =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, bodyA).xyz;
    const float3 linearVelocityB0 =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, bodyB).xyz;
    const float3 angularVelocityA0 =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, bodyA).xyz;
    const float3 angularVelocityB0 =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, bodyB).xyz;

    float3 linearVelocityA = linearVelocityA0;
    float3 linearVelocityB = linearVelocityB0;
    float3 angularVelocityA = angularVelocityA0;
    float3 angularVelocityB = angularVelocityB0;

    const uint storedContactCount = min(aggregateHeader.count, kRigidBodyPairAggregateContacts);
    [loop] for (uint solvePass = 0u; solvePass < 2u; ++solvePass)
    {
        [loop] for (uint contactOffset = 0u; contactOffset < kRigidBodyPairAggregateContacts;
                    ++contactOffset)
        {
            if (contactOffset >= storedContactCount)
            {
                break;
            }

            const uint flatSlotIndex =
                aggregateIndex * kRigidBodyPairAggregateContacts + contactOffset;
            GpuRigidBodyPairContactAggregateSlot slot =
                CRESSIM_SB_LOAD(g_RigidBodyPairAggregateSlots, flatSlotIndex);
            const bool hasRestitutionTarget = slot.solverState.y > 0.0;
            if ((solvePass == 0u && !hasRestitutionTarget) ||
                (solvePass == 1u && hasRestitutionTarget))
            {
                continue;
            }

            const float3 normal = SafeNormalize(slot.normal.xyz, float3(0.0, 1.0, 0.0));
            const float3 pointA =
                positionInvMassA.xyz + QuaternionRotate(orientationA, slot.localPointA.xyz);
            const float3 pointB =
                positionInvMassB.xyz + QuaternionRotate(orientationB, slot.localPointB.xyz);
            const float3 rA = pointA - positionInvMassA.xyz;
            const float3 rB = pointB - positionInvMassB.xyz;
            const float3 velocityA =
                ComputeContactPointVelocity(linearVelocityA, angularVelocityA, rA);
            const float3 velocityB =
                ComputeContactPointVelocity(linearVelocityB, angularVelocityB, rB);
            const float normalVelocity = dot(velocityB - velocityA, normal);
            const float normalDenom =
                ComputeContactEffectiveMass(invMassA, invInertiaA, orientationA, rA, normal) +
                ComputeContactEffectiveMass(invMassB, invInertiaB, orientationB, rB, normal);
            if (normalDenom <= kEpsilon)
            {
                continue;
            }

            const float deltaImpulse = (slot.solverState.y - normalVelocity) / normalDenom;
            const float newAccumulated = max(slot.solverState.x + deltaImpulse, 0.0);
            const float appliedImpulse = newAccumulated - slot.solverState.x;
            slot.solverState.x = newAccumulated;
            CRESSIM_SB_STORE(g_RigidBodyPairAggregateSlots, flatSlotIndex, slot);

            if (abs(appliedImpulse) <= kEpsilon)
            {
                continue;
            }

            const float3 impulse = normal * appliedImpulse;
            if (invMassA > kEpsilon)
            {
                linearVelocityA += -impulse * invMassA;
                angularVelocityA +=
                    MultiplyWorldInverseInertia(invInertiaA, orientationA, cross(rA, -impulse));
            }

            if (invMassB > kEpsilon)
            {
                linearVelocityB += impulse * invMassB;
                angularVelocityB +=
                    MultiplyWorldInverseInertia(invInertiaB, orientationB, cross(rB, impulse));
            }
        }
    }

    if (invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, bodyA,
                                      linearVelocityA - linearVelocityA0);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyAngularVelocityCorrections, bodyA,
                                      angularVelocityA - angularVelocityA0);
    }

    if (invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, bodyB,
                                      linearVelocityB - linearVelocityB0);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyAngularVelocityCorrections, bodyB,
                                      angularVelocityB - angularVelocityB0);
    }
}
