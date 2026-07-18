#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleStrandSegmentRanges);
CRESSIM_STRUCTURED_BUFFER(GpuStrandIncidentSegment, g_ParticleIncidentStrandSegments);
CRESSIM_STRUCTURED_BUFFER(GpuSoftEdgeCorrection, g_StrandDistanceCorrections);

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

    const GpuSoftConstraintRange range =
        CRESSIM_SB_LOAD(g_ParticleStrandSegmentRanges, particleIndex);
    float3 totalCorrection = float3(0.0, 0.0, 0.0);
    for (uint offset = 0u; offset < range.count; ++offset)
    {
        const GpuStrandIncidentSegment ref =
            CRESSIM_SB_LOAD(g_ParticleIncidentStrandSegments, range.start + offset);
        const GpuSoftEdgeCorrection correction =
            CRESSIM_SB_LOAD(g_StrandDistanceCorrections, ref.segmentIndex);
        totalCorrection += ref.slot == 0u ? correction.correctionA.xyz : correction.correctionB.xyz;
    }

    CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + totalCorrection, positionInvMass.w));
}
