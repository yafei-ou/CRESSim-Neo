#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

static const float kSoftCorrectionAtomicScale = 100000.0;
static const float kSoftContactRelaxation = 0.90;
static const float kSoftMaxCorrectionPerIter = 0.02;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftContact, g_SoftContacts);

CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftPositionCorrections);

int3 QuantizeSoftCorrection(float3 value)
{
    return int3(round(value * kSoftCorrectionAtomicScale));
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    if (contactIndex >= softCandidatePairCapacity)
    {
        return;
    }

    const GpuSoftContact contact = CRESSIM_SB_LOAD(g_SoftContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint softParticleIndex = contact.softParticleIndex;
    const float invMass = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, softParticleIndex).w;
    if (invMass <= kEpsilon)
    {
        return;
    }

    const float penetration =
        min(max(contact.normalPenetration.w - kContactSlop, 0.0), kSoftMaxCorrectionPerIter);
    if (penetration <= 0.0)
    {
        return;
    }

    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float3 correction = normal * (penetration * kSoftContactRelaxation);
    const int3 quantized = QuantizeSoftCorrection(correction);

    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, softParticleIndex).x, quantized.x);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, softParticleIndex).y, quantized.y);
    InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, softParticleIndex).z, quantized.z);
}
