#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

static const float kVelocityCorrectionAtomicScale = 100000.0;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticleVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftParticleVelocityCorrections);

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
        float3(CRESSIM_SB_REF(g_SoftParticleVelocityCorrections, particleIndex).xyz) /
        kVelocityCorrectionAtomicScale;

    if (invMass > kEpsilon)
    {
        velocity.xyz += correction;
    }

    CRESSIM_SB_STORE(g_SoftParticleVelocities, particleIndex, velocity);
    CRESSIM_SB_STORE(g_SoftParticleVelocityCorrections, particleIndex, int4(0, 0, 0, 0));
}
