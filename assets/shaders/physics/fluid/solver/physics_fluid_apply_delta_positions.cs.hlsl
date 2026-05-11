#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/core/physics_base.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

static const float kFluidRelaxationFactor = 0.85;

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidDeltaPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_PreviousFluidAccumulatedDeltaPositions);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount ||
        CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    if (positionInvMass.w <= kEpsilon)
    {
        return;
    }

    const float4 deltaAndWeight = CRESSIM_SB_LOAD(g_FluidDeltaPositions, particleIndex);
    const float weight = max(deltaAndWeight.w * kFluidRelaxationFactor, 1.0);
    const float3 delta = deltaAndWeight.xyz / weight;
    const float3 previousAccumulatedDelta =
        (iterationIndex > 0u)
            ? CRESSIM_SB_LOAD(g_PreviousFluidAccumulatedDeltaPositions, particleIndex).xyz
            : float3(0.0, 0.0, 0.0);
    CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + delta, positionInvMass.w));
    CRESSIM_SB_STORE(g_PreviousFluidAccumulatedDeltaPositions, particleIndex,
                     float4(previousAccumulatedDelta + delta, 0.0));
}
