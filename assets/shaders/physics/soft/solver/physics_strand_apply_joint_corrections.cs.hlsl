#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleStrandJointRanges);
CRESSIM_STRUCTURED_BUFFER(GpuStrandIncidentJoint, g_ParticleIncidentStrandJoints);
CRESSIM_STRUCTURED_BUFFER(GpuStrandJointCorrection, g_StrandJointCorrections);
CRESSIM_STRUCTURED_BUFFER(GpuStrandJoint, g_StrandJoints);
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
            const GpuSoftConstraintRange range = CRESSIM_SB_LOAD(g_ParticleStrandJointRanges, index);
            float3 totalCorrection = float3(0.0, 0.0, 0.0);
            for (uint offset = 0u; offset < range.count; ++offset)
            {
                const GpuStrandIncidentJoint ref =
                    CRESSIM_SB_LOAD(g_ParticleIncidentStrandJoints, range.start + offset);
                const GpuStrandJointCorrection correction =
                    CRESSIM_SB_LOAD(g_StrandJointCorrections, ref.jointIndex);
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
            CRESSIM_SB_STORE(g_ParticlePositionsInvMass, index,
                             float4(positionInvMass.xyz + totalCorrection, positionInvMass.w));
        }
    }

    if (index < strandJointCount)
    {
        const GpuStrandJoint joint = CRESSIM_SB_LOAD(g_StrandJoints, index);
        const GpuStrandJointCorrection correction =
            CRESSIM_SB_LOAD(g_StrandJointCorrections, index);
        GpuStrandSegmentState stateA;
        stateA.orientation = correction.orientationA;
        GpuStrandSegmentState stateB;
        stateB.orientation = correction.orientationB;
        CRESSIM_SB_STORE(g_StrandSegmentStates, joint.segmentA, stateA);
        CRESSIM_SB_STORE(g_StrandSegmentStates, joint.segmentB, stateB);
    }
}
