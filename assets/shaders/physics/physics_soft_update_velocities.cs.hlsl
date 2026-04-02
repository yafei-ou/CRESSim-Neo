#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePreviousPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticleVelocities);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= softParticleCount)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, idx);
    const float invMass = positionInvMass.w;
    if (invMass <= kEpsilon || dt <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_SoftParticleVelocities, idx, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float3 previousPosition = CRESSIM_SB_LOAD(g_SoftParticlePreviousPositions, idx).xyz;
    const float3 position = positionInvMass.xyz;
    CRESSIM_SB_STORE(g_SoftParticleVelocities, idx,
                     float4((position - previousPosition) / dt, 0.0));
}
