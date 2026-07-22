#include "physics_atomic_float.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_rigid_types.hlsli"
#include "physics_rigid_contact_primitives.hlsli"
#include "physics_rigid_solver_shared.hlsli"
#include "physics_math.hlsli"

// Soft-rigid position solve handles overlap and kinetic friction first.
// This pass is restitution-only and operates on reconstructed post-solve velocities.

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleContactMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuParticleRigidContact, g_ParticleRigidContacts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticleVelocityCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyAngularVelocityCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (contactIndex >= meta.activeParticleRigidContactCount)
    {
        return;
    }

    const GpuParticleRigidContact contact = CRESSIM_SB_LOAD(g_ParticleRigidContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint particleIndex = contact.particleIndex;
    const float4 softPositionInvMass =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float invMassSoft = softPositionInvMass.w;
    const float3 softVelocity = CRESSIM_SB_LOAD(g_ParticleVelocities, particleIndex).xyz;

    const uint rigidBodyIndex = contact.rigidBodyIndex;
    const uint rigidBodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndex);
    const float4 rigidPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex);
    const float invMassRigid =
        rigidBodyType == kRigidBodyTypeDynamic ? rigidPositionInvMass.w : 0.0;
    if (invMassSoft <= kEpsilon && invMassRigid <= kEpsilon)
    {
        return;
    }

    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
    const float3 rigidLinearVelocity =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyLinearVelocities, rigidBodyIndex).xyz;
    const float3 rigidAngularVelocity =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyAngularVelocities, rigidBodyIndex).xyz;
    const float3 invInertiaRigid =
        CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndex).xyz;

    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float3 rigidContactPoint =
        rigidPositionInvMass.xyz + QuaternionRotate(rigidOrientation, contact.rigidLocalPoint.xyz);
    const float3 rRigid = rigidContactPoint - rigidPositionInvMass.xyz;
    const float3 rigidContactVelocity = rigidLinearVelocity + cross(rigidAngularVelocity, rRigid);
    const float3 relativeVelocity = softVelocity - rigidContactVelocity;
    const float normalVelocity = dot(relativeVelocity, normal);
    if (normalVelocity >= 0.0)
    {
        return;
    }

    const float softNormalDenom = invMassSoft;
    const float rigidNormalDenom =
        ComputeContactEffectiveMass(invMassRigid, invInertiaRigid, rigidOrientation, rRigid, normal);
    const float normalDenom = softNormalDenom + rigidNormalDenom;
    if (normalDenom <= kEpsilon)
    {
        return;
    }

    const uint materialIndex = CRESSIM_SB_LOAD(g_ParticleMaterialIndices, particleIndex);
    const float3 combinedMaterial = CombineContactMaterial(
        CRESSIM_SB_LOAD(g_ParticleContactMaterials, materialIndex),
        CRESSIM_SB_LOAD(g_ColliderMaterials, contact.colliderIndex));
    const bool enableRestitution =
        (-normalVelocity > kRestitutionVelocityThreshold) &&
        (contact.normalPenetration.w <= kRestitutionPenetrationThreshold);
    const float restitution = enableRestitution ? saturate(combinedMaterial.y) : 0.0;
    const float desiredNormalVelocity =
        enableRestitution ? (-restitution * normalVelocity) : 0.0;
    const float normalImpulseScalar =
        max(0.0, (desiredNormalVelocity - normalVelocity) / normalDenom);
    const float3 totalImpulse = normal * normalImpulseScalar;

    if (invMassSoft > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticleVelocityCorrections, particleIndex,
                                      totalImpulse * invMassSoft);
    }

    if (invMassRigid > kEpsilon)
    {
        const float3 rigidLinearDelta = -totalImpulse * invMassRigid;
        const float3 rigidAngularDelta = MultiplyWorldInverseInertia(
            invInertiaRigid, rigidOrientation, cross(rRigid, -totalImpulse));
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyLinearVelocityCorrections, rigidBodyIndex,
                                      rigidLinearDelta);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyAngularVelocityCorrections, rigidBodyIndex,
                                      rigidAngularDelta);
    }
}
