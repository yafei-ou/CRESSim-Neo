#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_ParticleCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);

CRESSIM_RW_STRUCTURED_BUFFER(GpuParticleContact, g_ParticleContacts);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (pairIndex >= meta.particleParticleCandidateCount)
    {
        return;
    }

    GpuParticleContact outContact = (GpuParticleContact)0;

    const GpuParticleCandidatePair pair = CRESSIM_SB_LOAD(g_ParticleCandidatePairs, pairIndex);

    const uint particleA = pair.indexA;
    const uint particleB = pair.indexB;
    const uint kindA = CRESSIM_SB_LOAD(g_ParticleKinds, particleA);
    const uint kindB = CRESSIM_SB_LOAD(g_ParticleKinds, particleB);
    if (kindA == kParticleKindFluid && kindB == kParticleKindFluid)
    {
        CRESSIM_SB_STORE(g_ParticleContacts, pairIndex, outContact);
        CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, 0u);
        return;
    }
    const float3 positionA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleA).xyz;
    const float3 positionB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleB).xyz;
    const float radiusA = CRESSIM_SB_LOAD(g_ParticleRadii, particleA);
    const float radiusB = CRESSIM_SB_LOAD(g_ParticleRadii, particleB);
    const float3 delta = positionB - positionA;
    const float distanceSq = dot(delta, delta);
    const float combinedRadius = radiusA + radiusB;
    if (distanceSq <= kEpsilon)
    {
        outContact.particleA = particleA;
        outContact.particleB = particleB;
        outContact.active = 1u;
        outContact.normalPenetration = float4(0.0, 1.0, 0.0, combinedRadius);
        CRESSIM_SB_STORE(g_ParticleContacts, pairIndex, outContact);
        CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, 1u);
        return;
    }

    const float distance = sqrt(distanceSq);
    const float penetration = combinedRadius - distance;
    if (penetration > 0.0)
    {
        outContact.particleA = particleA;
        outContact.particleB = particleB;
        outContact.active = 1u;
        outContact.normalPenetration = float4(delta / distance, penetration);
    }

    CRESSIM_SB_STORE(g_ParticleContacts, pairIndex, outContact);
    CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, outContact.active);
}
