#include "physics/physics_atomic_float.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "physics/rigid/physics_rigid_solver_shared.hlsli"
#include "physics/core/physics_math.hlsli"

static const float kRestitutionVelocityThreshold = 0.5;
static const float kRestitutionPenetrationThreshold = 2.0 * kContactSlop;

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidProxyLocalPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuParticleContact, g_ParticleContacts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticleVelocityCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyAngularVelocityCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (contactIndex >= meta.activeParticleContactCount)
    {
        return;
    }

    const GpuParticleContact contact = CRESSIM_SB_LOAD(g_ParticleContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint particleA = contact.particleA;
    const uint particleB = contact.particleB;
    const float4 particleAState = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleA);
    const float4 particleBState = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleB);
    const uint ownerTypeA = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleA);
    const uint ownerTypeB = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleB);
    const bool proxyA = ownerTypeA == kParticleOwnerTypeRigidBody;
    const bool proxyB = ownerTypeB == kParticleOwnerTypeRigidBody;

    float invMassA = particleAState.w;
    float invMassB = particleBState.w;
    float3 velocityA = CRESSIM_SB_LOAD(g_ParticleVelocities, particleA).xyz;
    float3 velocityB = CRESSIM_SB_LOAD(g_ParticleVelocities, particleB).xyz;
    float3 rA = 0.0;
    float3 rB = 0.0;
    float3 invInertiaA = 0.0;
    float3 invInertiaB = 0.0;
    float4 orientationA = float4(0.0, 0.0, 0.0, 1.0);
    float4 orientationB = float4(0.0, 0.0, 0.0, 1.0);
    uint rigidBodyIndexA = 0u;
    uint rigidBodyIndexB = 0u;

    if (proxyA)
    {
        rigidBodyIndexA = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleA);
        const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndexA);
        const float4 rigidPositionInvMassA =
            CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndexA);
        orientationA =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndexA));
        invMassA = bodyTypeA == kRigidBodyTypeDynamic ? rigidPositionInvMassA.w : 0.0;
        invInertiaA = invMassA > kEpsilon
                          ? CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndexA).xyz
                          : 0.0;
        const float3 localProxyA = CRESSIM_SB_LOAD(g_RigidProxyLocalPositions, particleA).xyz;
        rA = QuaternionRotate(orientationA, localProxyA);
        velocityA = ComputeContactPointVelocity(
            CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, rigidBodyIndexA).xyz,
            CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, rigidBodyIndexA).xyz, rA);
    }

    if (proxyB)
    {
        rigidBodyIndexB = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleB);
        const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndexB);
        const float4 rigidPositionInvMassB =
            CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndexB);
        orientationB =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndexB));
        invMassB = bodyTypeB == kRigidBodyTypeDynamic ? rigidPositionInvMassB.w : 0.0;
        invInertiaB = invMassB > kEpsilon
                          ? CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndexB).xyz
                          : 0.0;
        const float3 localProxyB = CRESSIM_SB_LOAD(g_RigidProxyLocalPositions, particleB).xyz;
        rB = QuaternionRotate(orientationB, localProxyB);
        velocityB = ComputeContactPointVelocity(
            CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, rigidBodyIndexB).xyz,
            CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, rigidBodyIndexB).xyz, rB);
    }

    if (invMassA <= kEpsilon && invMassB <= kEpsilon)
    {
        return;
    }

    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float3 relativeVelocity = velocityB - velocityA;
    const float normalVelocity = dot(relativeVelocity, normal);
    if (normalVelocity >= 0.0)
    {
        return;
    }

    const float normalMassA = proxyA
                                  ? ComputeContactEffectiveMass(invMassA, invInertiaA, orientationA,
                                                                rA, normal)
                                  : invMassA;
    const float normalMassB = proxyB
                                  ? ComputeContactEffectiveMass(invMassB, invInertiaB, orientationB,
                                                                rB, normal)
                                  : invMassB;
    const float denom = normalMassA + normalMassB;
    if (denom <= kEpsilon)
    {
        return;
    }

    const float3 combinedMaterial = contact.material.xyz;
    const bool enableRestitution =
        (-normalVelocity > kRestitutionVelocityThreshold) &&
        (contact.normalPenetration.w <= kRestitutionPenetrationThreshold);
    const float restitution = enableRestitution ? saturate(combinedMaterial.y) : 0.0;
    const float desiredNormalVelocity =
        enableRestitution ? (-restitution * normalVelocity) : 0.0;
    const float normalImpulseScalar =
        max(0.0, (desiredNormalVelocity - normalVelocity) / denom);
    const float3 totalImpulse = normal * normalImpulseScalar;

    if (!proxyA && invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticleVelocityCorrections, particleA,
                                      -totalImpulse * invMassA);
    }

    if (!proxyB && invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticleVelocityCorrections, particleB,
                                      totalImpulse * invMassB);
    }

    if (proxyA && invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, rigidBodyIndexA,
                                      -totalImpulse * invMassA);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_RigidBodyAngularVelocityCorrections, rigidBodyIndexA,
            MultiplyWorldInverseInertia(invInertiaA, orientationA, cross(rA, -totalImpulse)));
    }

    if (proxyB && invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, rigidBodyIndexB,
                                      totalImpulse * invMassB);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(
            g_RigidBodyAngularVelocityCorrections, rigidBodyIndexB,
            MultiplyWorldInverseInertia(invInertiaB, orientationB, cross(rB, totalImpulse)));
    }
}
