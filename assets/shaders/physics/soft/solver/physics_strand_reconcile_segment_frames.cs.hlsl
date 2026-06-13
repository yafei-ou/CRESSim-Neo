#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_RW_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);

float4 ComputeReconciledOrientation(float3 delta, float4 previousOrientation)
{
    const float3 tangent = SafeNormalize(delta, float3(1.0, 0.0, 0.0));
    float3 normal = QuaternionRotate(previousOrientation, float3(0.0, 1.0, 0.0));
    normal = normal - tangent * dot(normal, tangent);
    if (dot(normal, normal) <= 1.0e-8)
    {
        normal = abs(tangent.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        normal -= tangent * dot(normal, tangent);
    }
    normal = SafeNormalize(normal, float3(0.0, 1.0, 0.0));
    const float3 binormal = SafeNormalize(cross(tangent, normal), float3(0.0, 0.0, 1.0));

    const float trace = tangent.x + normal.y + binormal.z;
    float4 q;
    if (trace > 0.0)
    {
        const float s = sqrt(trace + 1.0) * 2.0;
        q = float4((normal.z - binormal.y) / s, (binormal.x - tangent.z) / s,
                   (tangent.y - normal.x) / s, 0.25 * s);
    }
    else if (tangent.x > normal.y && tangent.x > binormal.z)
    {
        const float s = sqrt(1.0 + tangent.x - normal.y - binormal.z) * 2.0;
        q = float4(0.25 * s, (normal.x + tangent.y) / s, (binormal.x + tangent.z) / s,
                   (normal.z - binormal.y) / s);
    }
    else if (normal.y > binormal.z)
    {
        const float s = sqrt(1.0 + normal.y - tangent.x - binormal.z) * 2.0;
        q = float4((normal.x + tangent.y) / s, 0.25 * s, (binormal.y + normal.z) / s,
                   (binormal.x - tangent.z) / s);
    }
    else
    {
        const float s = sqrt(1.0 + binormal.z - tangent.x - normal.y) * 2.0;
        q = float4((binormal.x + tangent.z) / s, (binormal.y + normal.z) / s, 0.25 * s,
                   (tangent.y - normal.x) / s);
    }
    return QuaternionNormalize(q);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint segmentIndex = dispatchThreadID.x;
    if (segmentIndex >= strandSegmentCount)
    {
        return;
    }

    const GpuStrandSegment segment = CRESSIM_SB_LOAD(g_StrandSegments, segmentIndex);
    const float3 positionA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleA).xyz;
    const float3 positionB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleB).xyz;
    const GpuStrandSegmentState previousState = CRESSIM_SB_LOAD(g_StrandSegmentStates, segmentIndex);
    GpuStrandSegmentState nextState;
    nextState.orientation = ComputeReconciledOrientation(positionB - positionA, previousState.orientation);
    CRESSIM_SB_STORE(g_StrandSegmentStates, segmentIndex, nextState);
}
