#include "include/physics/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_SoftParticleRadii);
CRESSIM_STRUCTURED_BUFFER(GpuSoftCandidatePair, g_SoftCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);

CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftContact, g_SoftContacts);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_ContactActiveFlags);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    const GpuSoftNeighborMeta meta = CRESSIM_SB_LOAD(g_SoftNeighborMeta, 0u);
    if (pairIndex >= meta.softSoftCandidateCount)
    {
        return;
    }

    GpuSoftContact outContact;
    outContact.particleA = 0u;
    outContact.particleB = 0u;
    outContact.active = 0u;
    outContact.reserved0 = 0u;
    outContact.normalPenetration = float4(0.0, 0.0, 0.0, 0.0);

    const GpuSoftCandidatePair pair = CRESSIM_SB_LOAD(g_SoftCandidatePairs, pairIndex);

    const uint particleA = pair.indexA;
    const uint particleB = pair.indexB;
    const float3 positionA = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleA).xyz;
    const float3 positionB = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleB).xyz;
    const float radiusA = CRESSIM_SB_LOAD(g_SoftParticleRadii, particleA);
    const float radiusB = CRESSIM_SB_LOAD(g_SoftParticleRadii, particleB);
    const float3 delta = positionB - positionA;
    const float distanceSq = dot(delta, delta);
    const float combinedRadius = radiusA + radiusB;
    if (distanceSq <= kEpsilon)
    {
        outContact.particleA = particleA;
        outContact.particleB = particleB;
        outContact.active = 1u;
        outContact.normalPenetration = float4(0.0, 1.0, 0.0, combinedRadius);
        CRESSIM_SB_STORE(g_SoftContacts, pairIndex, outContact);
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

    CRESSIM_SB_STORE(g_SoftContacts, pairIndex, outContact);
    CRESSIM_SB_STORE(g_ContactActiveFlags, pairIndex, outContact.active);
}
