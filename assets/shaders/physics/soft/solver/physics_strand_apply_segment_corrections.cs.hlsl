#include "physics/physics_particle_dispatch_constants.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/core/physics_math.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleStrandSegmentRanges);
CRESSIM_STRUCTURED_BUFFER(GpuStrandIncidentSegment, g_ParticleIncidentStrandSegments);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegmentCorrection, g_StrandSegmentCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index < particleCount)
    {
        const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, index);
        if (positionInvMass.w > kEpsilon)
        {
            const GpuSoftConstraintRange range = CRESSIM_SB_LOAD(g_ParticleStrandSegmentRanges, index);
            float3 totalCorrection = float3(0.0, 0.0, 0.0);
            for (uint offset = 0u; offset < range.count; ++offset)
            {
                const GpuStrandIncidentSegment ref =
                    CRESSIM_SB_LOAD(g_ParticleIncidentStrandSegments, range.start + offset);
                const GpuStrandSegmentCorrection correction =
                    CRESSIM_SB_LOAD(g_StrandSegmentCorrections, ref.segmentIndex);
                totalCorrection += ref.slot == 0u ? correction.correctionA.xyz : correction.correctionB.xyz;
            }
            CRESSIM_SB_STORE(g_ParticlePositionsInvMass, index,
                             float4(positionInvMass.xyz + totalCorrection, positionInvMass.w));
        }
    }

    if (index < strandSegmentCount)
    {
        const GpuStrandSegmentCorrection correction =
            CRESSIM_SB_LOAD(g_StrandSegmentCorrections, index);
        GpuStrandSegmentState state = CRESSIM_SB_LOAD(g_StrandSegmentStates, index);
        state.orientation = QuaternionNormalize(state.orientation + correction.angularCorrection);
        CRESSIM_SB_STORE(g_StrandSegmentStates, index, state);
    }
}
