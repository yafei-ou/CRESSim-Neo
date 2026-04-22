#include "include/physics/physics_rigid_common.hlsli"
#include "include/physics/physics_soft_dispatch_constants.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftConstraintRange, g_ParticleEdgeRanges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftIncidentEdge, g_ParticleIncidentEdges);
CRESSIM_STRUCTURED_BUFFER(GpuSoftEdgeCorrection, g_SoftEdgeCorrections);

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

    const GpuSoftConstraintRange range = CRESSIM_SB_LOAD(g_ParticleEdgeRanges, particleIndex);
    float3 totalCorrection = float3(0.0, 0.0, 0.0);
    for (uint offset = 0u; offset < range.count; ++offset)
    {
        const GpuSoftIncidentEdge ref = CRESSIM_SB_LOAD(g_ParticleIncidentEdges, range.start + offset);
        const GpuSoftEdgeCorrection correction = CRESSIM_SB_LOAD(g_SoftEdgeCorrections, ref.edgeIndex);
        totalCorrection += ref.slot == 0u ? correction.correctionA.xyz : correction.correctionB.xyz;
    }

    CRESSIM_SB_STORE(g_SoftParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + totalCorrection, positionInvMass.w));
}
