#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuStrandJointCorrection, g_StrandJointCorrections);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_SegmentStrandJointRanges);
CRESSIM_STRUCTURED_BUFFER(GpuStrandIncidentJoint, g_SegmentIncidentStrandJoints);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index < strandSegmentCount)
    {
        const GpuSoftConstraintRange range = CRESSIM_SB_LOAD(g_SegmentStrandJointRanges, index);
        GpuStrandSegmentState state = CRESSIM_SB_LOAD(g_StrandSegmentStates, index);
        float4 totalOrientationCorrection = float4(0.0, 0.0, 0.0, 0.0);
        for (uint offset = 0u; offset < range.count; ++offset)
        {
            const GpuStrandIncidentJoint ref =
                CRESSIM_SB_LOAD(g_SegmentIncidentStrandJoints, range.start + offset);
            const GpuStrandJointCorrection correction =
                CRESSIM_SB_LOAD(g_StrandJointCorrections, ref.jointIndex);
            totalOrientationCorrection +=
                ref.slot == 0u ? correction.twistRotationA : correction.twistRotationB;
        }

        state.orientation = QuaternionNormalize(state.orientation + totalOrientationCorrection);
        CRESSIM_SB_STORE(g_StrandSegmentStates, index, state);
    }
}
