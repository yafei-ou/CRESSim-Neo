#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticleVelocities);
CRESSIM_RW_BYTE_ADDRESS_BUFFER(g_SoftParticleVelocityCorrections);

// Mirrors the rigid contact-velocity apply step: fold accumulated soft particle velocity
// corrections back into the live velocity buffer between iterations.

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= softParticleCount)
    {
        return;
    }
    float4 velocity = CRESSIM_SB_LOAD(g_SoftParticleVelocities, particleIndex);
    const float invMass = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleIndex).w;
    const float3 correction =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_SoftParticleVelocityCorrections, particleIndex);

    if (invMass > kEpsilon)
    {
        velocity.xyz += correction;
    }

    CRESSIM_SB_STORE(g_SoftParticleVelocities, particleIndex, velocity);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_SoftParticleVelocityCorrections, particleIndex);
}
