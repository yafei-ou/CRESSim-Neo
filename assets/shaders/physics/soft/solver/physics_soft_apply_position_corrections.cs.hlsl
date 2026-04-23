#include "../../../include/physics/physics_soft_dispatch_constants.hlsli"
#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/core/physics_base.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_SoftPositionCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= softParticleCount)
    {
        return;
    }

    const float3 correction =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_SoftPositionCorrections, particleIndex);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_SoftPositionCorrections, particleIndex);

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleIndex);
    const float invMass = positionInvMass.w;
    if (invMass <= kEpsilon)
    {
        return;
    }

    CRESSIM_SB_STORE(g_SoftParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + correction, invMass));
}
