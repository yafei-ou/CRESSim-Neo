#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleBendRanges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftIncidentBend, g_ParticleIncidentBends);
CRESSIM_STRUCTURED_BUFFER(GpuSoftBendCorrection, g_SoftBendCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    if (positionInvMass.w <= kEpsilon)
    {
        return;
    }

    const GpuSoftConstraintRange range = CRESSIM_SB_LOAD(g_ParticleBendRanges, particleIndex);
    float3 totalCorrection = float3(0.0, 0.0, 0.0);
    for (uint offset = 0u; offset < range.count; ++offset)
    {
        const GpuSoftIncidentBend ref =
            CRESSIM_SB_LOAD(g_ParticleIncidentBends, range.start + offset);
        const GpuSoftBendCorrection correction =
            CRESSIM_SB_LOAD(g_SoftBendCorrections, ref.bendIndex);
        if (ref.slot == 0u)
        {
            totalCorrection += correction.correction0.xyz;
        }
        else if (ref.slot == 1u)
        {
            totalCorrection += correction.correction1.xyz;
        }
        else
        {
            totalCorrection += correction.correction2.xyz;
        }
    }

    CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + totalCorrection, positionInvMass.w));
}
