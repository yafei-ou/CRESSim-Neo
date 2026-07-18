#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(float4, g_FluidIterationDelta);
CRESSIM_STRUCTURED_BUFFER(uint2, g_FluidBoundaryCandidateRanges);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidBoundaryCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);
CRESSIM_STRUCTURED_BUFFER(GpuColliderGeometryData, g_ColliderGeometryData);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_FluidIterationDeltaRW);

#include "physics/fluid/physics_fluid_common.hlsli"
#include "physics/fluid/physics_fluid_boundary_common.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount ||
        CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        return;
    }

    const float3 basePosition = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const float particleRadius = max(CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex), 0.0);
    float3 currentPosition =
        basePosition + CRESSIM_SB_LOAD(g_FluidIterationDelta, particleIndex).xyz;

    const uint2 candidateRange = CRESSIM_SB_LOAD(g_FluidBoundaryCandidateRanges, particleIndex);
    const uint candidateOffset = candidateRange.x;
    const uint candidateCount = candidateRange.y;

    [loop]
    for (uint sweep = 0u; sweep < 2u; ++sweep)
    {
        bool changed = false;

        [loop]
        for (uint i = 0u; i < candidateCount; ++i)
        {
            const GpuParticleCandidatePair pair =
                CRESSIM_SB_LOAD(g_FluidBoundaryCandidatePairs, candidateOffset + i);
            const uint colliderIndex = pair.indexB;
            const uint rigidBodyIndex = pair.auxIndex;

            float signedDistance = 0.0;
            float3 normalWorld = float3(0.0, 1.0, 0.0);
            if (!ComputeFluidBoundaryContactInfo(rigidBodyIndex, colliderIndex, currentPosition,
                                                 signedDistance, normalWorld))
            {
                continue;
            }

            if (signedDistance >= particleRadius)
            {
                continue;
            }

            currentPosition += normalWorld * (particleRadius - signedDistance);
            changed = true;
        }

        if (!changed)
        {
            break;
        }
    }

    CRESSIM_SB_STORE(g_FluidIterationDeltaRW, particleIndex,
                     float4(currentPosition - basePosition, 0.0));
}
