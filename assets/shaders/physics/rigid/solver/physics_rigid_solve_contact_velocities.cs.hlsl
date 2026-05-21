#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "../../../include/physics/physics_rigid_dispatch_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRigidContact, g_RigidContacts);
CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);

CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidContactVelocityState, g_RigidContactVelocityStates);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyAngularVelocityCorrections);

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

    const GpuRigidContact contact = CRESSIM_SB_LOAD(g_RigidContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint bodyA = contact.bodyA;
    const uint bodyB = contact.bodyB;
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
    const float3 normal = SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float normalVelocity = dot(velocityB - velocityA, normal);
    const float normalDenom =
        ComputeContactEffectiveMass(invMassA, invInertiaA, orientationA, rA, normal) +
        ComputeContactEffectiveMass(invMassB, invInertiaB, orientationB, rB, normal);
    if (normalDenom <= kEpsilon)
    {
        return;
    }

    GpuRigidContactVelocityState state =
        CRESSIM_SB_LOAD(g_RigidContactVelocityStates, contactIndex);
    const float deltaImpulse = (state.targetNormalVelocity - normalVelocity) / normalDenom;
    const float newAccumulated = max(state.accumulatedNormalImpulse + deltaImpulse, 0.0);
    const float appliedImpulse = newAccumulated - state.accumulatedNormalImpulse;
    state.accumulatedNormalImpulse = newAccumulated;
    CRESSIM_SB_STORE(g_RigidContactVelocityStates, contactIndex, state);

    if (abs(appliedImpulse) <= kEpsilon)
    {
        return;
    }

    const float3 impulse = normal * appliedImpulse;
    if (invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, bodyA,
                                      -impulse * invMassA);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_RigidBodyAngularVelocityCorrections, bodyA,
            MultiplyWorldInverseInertia(invInertiaA, orientationA, cross(rA, -impulse)));
    }

    if (invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, bodyB,
                                      impulse * invMassB);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_RigidBodyAngularVelocityCorrections, bodyB,
            MultiplyWorldInverseInertia(invInertiaB, orientationB, cross(rB, impulse)));
    }
}
