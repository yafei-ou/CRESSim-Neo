#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

static const float kSoftCorrectionAtomicScale = 100000.0;

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftPositionCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= softParticleCount)
    {
        return;
    }

    const int4 accumulatedCorrection = CRESSIM_SB_REF(g_SoftPositionCorrections, particleIndex);
    CRESSIM_SB_STORE(g_SoftPositionCorrections, particleIndex, int4(0, 0, 0, 0));

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleIndex);
    const float invMass = positionInvMass.w;
    if (invMass <= kEpsilon)
    {
        return;
    }

    const float3 correction = float3(accumulatedCorrection.xyz) / kSoftCorrectionAtomicScale;
    CRESSIM_SB_STORE(g_SoftParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + correction, invMass));
}
