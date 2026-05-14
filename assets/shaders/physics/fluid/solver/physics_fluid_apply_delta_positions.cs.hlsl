#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/core/physics_base.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/fluid/physics_fluid_common.hlsli"

static const float kFluidRelaxationFactor = 1.0;

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidDeltaPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidIterationDelta);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount ||
        CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        return;
    }

    const float4 deltaAndWeight = CRESSIM_SB_LOAD(g_FluidDeltaPositions, particleIndex);
    const float weight = max(deltaAndWeight.w * kFluidRelaxationFactor, 1.0);
    const float3 delta = deltaAndWeight.xyz / weight;
    const float3 basePosition =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const float particleRadius = max(CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex), 0.0);
    const float4 previousAccumulatedDeltaAndWeight =
        CRESSIM_SB_LOAD(g_FluidIterationDelta, particleIndex);
    const float3 previousAccumulatedDelta =
        (iterationIndex > 0u) ? previousAccumulatedDeltaAndWeight.xyz : float3(0.0, 0.0, 0.0);
    const float3 unclampedAccumulatedDelta = previousAccumulatedDelta + delta;
    const float3 clampedPosition =
        ClampManualFluidBoundaryPosition(basePosition + unclampedAccumulatedDelta, particleRadius);
    CRESSIM_SB_STORE(g_FluidIterationDelta, particleIndex,
                     float4(clampedPosition - basePosition, 0.0));
}
