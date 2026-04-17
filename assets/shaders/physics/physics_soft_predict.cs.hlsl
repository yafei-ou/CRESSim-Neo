#include "physics/include/physics_soft_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticlePreviousPositions);
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
    const float3 position = positionInvMass.xyz;
    const float invMass = positionInvMass.w;

    CRESSIM_SB_STORE(g_SoftParticlePreviousPositions, idx, float4(position, 0.0));

    if (invMass <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_SoftParticleVelocities, idx, float4(0.0, 0.0, 0.0, 0.0));
        CRESSIM_SB_STORE(g_SoftParticlePositionsInvMass, idx, positionInvMass);
        return;
    }

    float3 velocity = CRESSIM_SB_LOAD(g_SoftParticleVelocities, idx).xyz;
    velocity += kGravity * dt;
    const float3 predictedPosition = position + velocity * dt;

    CRESSIM_SB_STORE(g_SoftParticleVelocities, idx, float4(velocity, 0.0));
    CRESSIM_SB_STORE(g_SoftParticlePositionsInvMass, idx, float4(predictedPosition, invMass));
}
