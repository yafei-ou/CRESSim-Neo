#include "physics_particle_dispatch_constants.hlsli"
#include "physics_base.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleContactMaterials);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidIterationDelta);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;
    if (idx >= particleCount)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, idx);
    const float invMass = positionInvMass.w;
    if (invMass <= kEpsilon || dt <= kEpsilon)
    {
        CRESSIM_SB_STORE(g_ParticleVelocities, idx, float4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    const float3 previousPosition = CRESSIM_SB_LOAD(g_ParticlePreviousPositions, idx).xyz;
    float3 position = positionInvMass.xyz;
    if (CRESSIM_SB_LOAD(g_ParticleKinds, idx) == kParticleKindFluid)
    {
        position += CRESSIM_SB_LOAD(g_FluidIterationDelta, idx).xyz;
        CRESSIM_SB_STORE(g_ParticlePositionsInvMass, idx, float4(position, positionInvMass.w));
        CRESSIM_SB_STORE(g_FluidIterationDelta, idx,
                         float4(0.0, 0.0, 0.0, 0.0));
    }
    const uint materialIndex = CRESSIM_SB_LOAD(g_ParticleMaterialIndices, idx);
    const float damping = max(CRESSIM_SB_LOAD(g_ParticleContactMaterials, materialIndex).z, 0.0);
    const float dampingScale = max(1.0 - damping * dt, 0.0);
    CRESSIM_SB_STORE(g_ParticleVelocities, idx,
                     float4(((position - previousPosition) / dt) * dampingScale, 0.0));
}
