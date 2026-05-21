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

CRESSIM_RW_STRUCTURED_BUFFER(GpuRigidContactVelocityState, g_RigidContactVelocityStates);

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

    GpuRigidContactVelocityState state;
    state.accumulatedNormalImpulse = 0.0;
    state.targetNormalVelocity = 0.0;
    state.reserved0 = 0.0;
    state.reserved1 = 0.0;

    const GpuRigidContact contact = CRESSIM_SB_LOAD(g_RigidContacts, contactIndex);
    if (contact.active != 0u)
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
        const float normalVelocity = dot(velocityB - velocityA, normal);
        if (normalVelocity < -kRigidRestitutionThreshold)
        {
            const float restitution = saturate(contact.material.y);
            state.targetNormalVelocity = -restitution * normalVelocity;
        }
    }

    CRESSIM_SB_STORE(g_RigidContactVelocityStates, contactIndex, state);
}
