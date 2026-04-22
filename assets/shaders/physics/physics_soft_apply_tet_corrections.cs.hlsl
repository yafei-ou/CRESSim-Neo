#include "include/physics/physics_rigid_common.hlsli"
#include "include/physics/physics_soft_dispatch_constants.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleTetRanges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftIncidentTet, g_ParticleIncidentTets);
CRESSIM_STRUCTURED_BUFFER(GpuSoftTetCorrection, g_SoftTetCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= softParticleCount)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, particleIndex);
    if (positionInvMass.w <= kEpsilon)
    {
        return;
    }

    const GpuSoftConstraintRange range = CRESSIM_SB_LOAD(g_ParticleTetRanges, particleIndex);
    float3 totalCorrection = float3(0.0, 0.0, 0.0);
    for (uint offset = 0u; offset < range.count; ++offset)
    {
        const GpuSoftIncidentTet ref = CRESSIM_SB_LOAD(g_ParticleIncidentTets, range.start + offset);
        const GpuSoftTetCorrection correction = CRESSIM_SB_LOAD(g_SoftTetCorrections, ref.tetIndex);
        if (ref.slot == 0u)
        {
            totalCorrection += correction.correction0.xyz;
        }
        else if (ref.slot == 1u)
        {
            totalCorrection += correction.correction1.xyz;
        }
        else if (ref.slot == 2u)
        {
            totalCorrection += correction.correction2.xyz;
        }
        else
        {
            totalCorrection += correction.correction3.xyz;
        }
    }

    CRESSIM_SB_STORE(g_SoftParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + totalCorrection, positionInvMass.w));
}
