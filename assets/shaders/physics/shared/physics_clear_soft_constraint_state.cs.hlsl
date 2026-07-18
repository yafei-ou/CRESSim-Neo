#include "physics_particle_dispatch_constants.hlsli"
#include "physics_atomic_float.hlsli"
#include "structured_buffer_compat.hlsli"

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticleVelocityCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftBendLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftTetLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_StrandSegmentLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_StrandJointLambdas);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_StrandDistanceLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadID.x;

    if (idx < particleCount)
    {
        CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_ParticlePositionCorrections, idx);
        CRESSIM_CLEAR_ATOMIC_FLOAT4_ENTRY(g_ParticleVelocityCorrections, idx);
    }

    if (idx < softEdgeCount)
    {
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, idx, 0.0);
    }

    if (idx < softBendCount)
    {
        CRESSIM_SB_STORE(g_SoftBendLambdas, idx, 0.0);
    }

    if (idx < softTetCount)
    {
        CRESSIM_SB_STORE(g_SoftTetLambdas, idx, 0.0);
    }

    if (idx < strandSegmentCount)
    {
        CRESSIM_SB_STORE(g_StrandSegmentLambdas, idx, float4(0.0, 0.0, 0.0, 0.0));
    }

    if (idx < strandJointCount)
    {
        CRESSIM_SB_STORE(g_StrandJointLambdas, idx, float4(0.0, 0.0, 0.0, 0.0));
    }

    if (idx < strandDistanceCount)
    {
        CRESSIM_SB_STORE(g_StrandDistanceLambdas, idx, 0.0);
    }
}
