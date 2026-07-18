#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidProxyLocalPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    if (CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleIndex) != kParticleOwnerTypeRigidBody)
    {
        return;
    }

    const uint rigidBodyIndex = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleIndex);
    const float3 localProxy = CRESSIM_SB_LOAD(g_RigidProxyLocalPositions, particleIndex).xyz;
    const float4 orientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
    const float3 bodyPosition =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex).xyz;
    const float3 proxyWorldPosition =
        bodyPosition + QuaternionRotate(orientation, localProxy);

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex,
                     float4(proxyWorldPosition, positionInvMass.w));
}
