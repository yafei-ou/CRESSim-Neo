#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/core/physics_base.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= particleCount)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, idx);
    const float3 position = positionInvMass.xyz;
    const float invMass = positionInvMass.w;

    CRESSIM_SB_STORE(g_ParticlePreviousPositions, idx, float4(position, 0.0));

    if (invMass <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_ParticleVelocities, idx, float4(0.0, 0.0, 0.0, 0.0));
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, idx, positionInvMass);
        return;
    }

    float3 velocity = CRESSIM_SB_LOAD(g_ParticleVelocities, idx).xyz;
    float gravityScale = 1.0;
    if (CRESSIM_SB_LOAD(g_ParticleKinds, idx) == kParticleKindFluid)
    {
        gravityScale = fluidGravityScale;
    }
    velocity += kGravity * (dt * gravityScale);
    const float3 predictedPosition = position + velocity * dt;

    CRESSIM_SB_STORE(g_ParticleVelocities, idx, float4(velocity, 0.0));
    CRESSIM_SB_STORE(g_ParticlePositionsInvMass, idx, float4(predictedPosition, invMass));
}
