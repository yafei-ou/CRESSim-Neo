#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuStrandRigidAttachmentCorrection, g_StrandRigidAttachmentCorrections);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_SegmentStrandRigidAttachmentRanges);
CRESSIM_STRUCTURED_BUFFER(GpuStrandIncidentAttachment, g_SegmentIncidentStrandRigidAttachments);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint segmentIndex = dispatchThreadID.x;
    if (segmentIndex >= strandSegmentCount)
    {
        return;
    }

    const GpuSoftConstraintRange range =
        CRESSIM_SB_LOAD(g_SegmentStrandRigidAttachmentRanges, segmentIndex);
    float3 totalRotation = 0.0;
    for (uint offset = 0u; offset < range.count; ++offset)
    {
        const GpuStrandIncidentAttachment ref =
            CRESSIM_SB_LOAD(g_SegmentIncidentStrandRigidAttachments, range.start + offset);
        const GpuStrandRigidAttachmentCorrection correction =
            CRESSIM_SB_LOAD(g_StrandRigidAttachmentCorrections, ref.attachmentIndex);
        totalRotation += correction.segmentRotation.xyz;
    }

    if (dot(totalRotation, totalRotation) <= kEpsilon * kEpsilon)
    {
        return;
    }

    GpuStrandSegmentState state = CRESSIM_SB_LOAD(g_StrandSegmentStates, segmentIndex);
    state.orientation = QuaternionNormalize(
        QuaternionMul(QuaternionFromRotationVector(totalRotation), state.orientation));
    CRESSIM_SB_STORE(g_StrandSegmentStates, segmentIndex, state);
}
