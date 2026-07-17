#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/physics_atomic_float.hlsli"
#include "physics/core/physics_base.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    const float3 correction =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_ParticlePositionCorrections, particleIndex);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_ParticlePositionCorrections, particleIndex);

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float invMass = positionInvMass.w;
    if (invMass <= kEpsilon)
    {
        return;
    }

    CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + correction, invMass));
}
