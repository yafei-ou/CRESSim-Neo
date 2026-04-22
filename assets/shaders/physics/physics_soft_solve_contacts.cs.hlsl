#include "include/physics/physics_rigid_common.hlsli"

static const float kSoftContactRelaxation = 0.95;
static const float kSoftMaxCorrectionPerIter = 0.05;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePreviousPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticleMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuSoftContact, g_SoftContacts);
CRESSIM_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_SoftPositionCorrections);

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
    const float3 previousPositionA =
        CRESSIM_SB_LOAD(g_SoftParticlePreviousPositions, particleA).xyz;
    const float3 previousPositionB =
        CRESSIM_SB_LOAD(g_SoftParticlePreviousPositions, particleB).xyz;
    const float invMassA = particleAState.w;
    const float invMassB = particleBState.w;
    if (invMassA <= kEpsilon && invMassB <= kEpsilon)
    {
        return;
    }

    const float penetration =
        min(max(contact.normalPenetration.w - kContactSlop, 0.0), kSoftMaxCorrectionPerIter);
    if (penetration <= 0.0)
    {
        return;
    }

    const float3 combinedMaterial = CombineContactMaterial(
        CRESSIM_SB_LOAD(g_SoftParticleMaterials, particleA),
        CRESSIM_SB_LOAD(g_SoftParticleMaterials, particleB));
    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float3 relativeDisplacement =
        (particleBState.xyz - previousPositionB) - (particleAState.xyz - previousPositionA);
    const float denom = invMassA + invMassB;
    if (denom <= kEpsilon)
    {
        return;
    }

    const float lambda = (penetration / denom) * kSoftContactRelaxation;
    float3 correctionA = -normal * (invMassA * lambda);
    float3 correctionB = normal * (invMassB * lambda);

    const float3 tangentialDisplacement = ProjectOntoContactTangent(relativeDisplacement, normal);
    const float3 frictionDelta = ComputePositionFrictionDelta(
        tangentialDisplacement, penetration, combinedMaterial.x, combinedMaterial.z);
    if (length(frictionDelta) > 0.0)
    {
        correctionA += frictionDelta * (invMassA / denom) * kSoftContactRelaxation;
        correctionB -= frictionDelta * (invMassB / denom) * kSoftContactRelaxation;
    }

    if (invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_SoftPositionCorrections, particleA, correctionA);
    }

    if (invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_SoftPositionCorrections, particleB, correctionB);
    }
}
