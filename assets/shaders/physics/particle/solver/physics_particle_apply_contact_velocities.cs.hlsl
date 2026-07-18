#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/physics_atomic_float.hlsli"
#include "physics/core/physics_base.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticleVelocityCorrections);

// Mirrors the rigid contact-velocity apply step: fold accumulated soft particle velocity
// corrections back into the live velocity buffer between iterations.

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }
    float4 velocity = CRESSIM_SB_LOAD(g_ParticleVelocities, particleIndex);
    const float invMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).w;
    const float3 correction =
        CRESSIM_LOAD_ATOMIC_FLOAT3_ENTRY(g_ParticleVelocityCorrections, particleIndex);

    if (invMass > kEpsilon)
    {
        velocity.xyz += correction;
    }

    CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, velocity);
    CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_ParticleVelocityCorrections, particleIndex);
}
