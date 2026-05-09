#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"

static const float kSoftContactRelaxation = 0.95;
static const float kSoftMaxCorrectionPerIter = 0.05;

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuParticleContact, g_ParticleContacts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);

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
    const float3 previousPositionA =
        CRESSIM_SB_LOAD(g_ParticlePreviousPositions, particleA).xyz;
    const float3 previousPositionB =
        CRESSIM_SB_LOAD(g_ParticlePreviousPositions, particleB).xyz;
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
        CRESSIM_SB_LOAD(g_ParticleMaterials, particleA),
        CRESSIM_SB_LOAD(g_ParticleMaterials, particleB));
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
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, particleA, correctionA);
    }

    if (invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, particleB, correctionB);
    }
}
