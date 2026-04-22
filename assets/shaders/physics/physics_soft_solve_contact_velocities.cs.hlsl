#include "include/physics/physics_rigid_common.hlsli"

static const float kRestitutionVelocityThreshold = 0.5;
static const float kRestitutionPenetrationThreshold = 2.0 * kContactSlop;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticleMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticleVelocities);
CRESSIM_STRUCTURED_BUFFER(GpuSoftContact, g_SoftContacts);
CRESSIM_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_SoftParticleVelocityCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const GpuSoftNeighborMeta meta = CRESSIM_SB_LOAD(g_SoftNeighborMeta, 0u);
    if (contactIndex >= meta.activeSoftContactCount)
    {
        return;
    }

    const GpuSoftContact contact = CRESSIM_SB_LOAD(g_SoftContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint particleA = contact.particleA;
    const uint particleB = contact.particleB;
    const float4 particleAState = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleA);
    const float4 particleBState = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleB);
    const float invMassA = particleAState.w;
    const float invMassB = particleBState.w;
    if (invMassA <= kEpsilon && invMassB <= kEpsilon)
    {
        return;
    }

    const float3 velocityA = CRESSIM_SB_LOAD(g_SoftParticleVelocities, particleA).xyz;
    const float3 velocityB = CRESSIM_SB_LOAD(g_SoftParticleVelocities, particleB).xyz;
    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float3 relativeVelocity = velocityB - velocityA;
    const float normalVelocity = dot(relativeVelocity, normal);
    if (normalVelocity >= 0.0)
    {
        return;
    }

    const float denom = invMassA + invMassB;
    if (denom <= kEpsilon)
    {
        return;
    }

    const float3 combinedMaterial = CombineContactMaterial(
        CRESSIM_SB_LOAD(g_SoftParticleMaterials, particleA),
        CRESSIM_SB_LOAD(g_SoftParticleMaterials, particleB));
    const bool enableRestitution =
        (-normalVelocity > kRestitutionVelocityThreshold) &&
        (contact.normalPenetration.w <= kRestitutionPenetrationThreshold);
    const float restitution = enableRestitution ? saturate(combinedMaterial.y) : 0.0;
    const float desiredNormalVelocity =
        enableRestitution ? (-restitution * normalVelocity) : 0.0;
    const float normalImpulseScalar =
        max(0.0, (desiredNormalVelocity - normalVelocity) / denom);
    const float3 totalImpulse = normal * normalImpulseScalar;

    if (invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_SoftParticleVelocityCorrections, particleA,
                                      -totalImpulse * invMassA);
    }

    if (invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_SoftParticleVelocityCorrections, particleB,
                                      totalImpulse * invMassB);
    }
}
