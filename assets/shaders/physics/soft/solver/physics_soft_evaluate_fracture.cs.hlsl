#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_RW_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);
CRESSIM_RW_STRUCTURED_BUFFER(float, g_SoftEdgeLambdas);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint edgeIndex = dispatchThreadID.x;
    if (edgeIndex >= softEdgeCount)
    {
        return;
    }

    GpuSoftEdge edge = CRESSIM_SB_LOAD(g_SoftEdges, edgeIndex);
    if ((edge.flags & kSoftEdgeActiveFlag) == 0u ||
        (edge.flags & kSoftEdgeDisabledFlag) != 0u)
    {
        CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
        return;
    }

    const float3 positionA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleA).xyz;
    const float3 positionB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, edge.particleB).xyz;
    const float currentLength = length(positionB - positionA);
    const float restLength = max(edge.restLength, kEpsilon);
    edge.strain = (currentLength - edge.restLength) / restLength;

    const float strainMagnitude = abs(edge.strain);
    if (edge.failureThreshold > 0.0 && strainMagnitude > edge.failureThreshold)
    {
        const float excess = strainMagnitude - edge.failureThreshold;
        edge.damage = saturate(edge.damage + excess * dt / max(edge.cutResistance, kEpsilon));
        if (edge.damage >= 1.0)
        {
            edge.flags = (edge.flags | kSoftEdgeFracturedFlag | kSoftEdgeDisabledFlag) &
                         ~kSoftEdgeActiveFlag;
            CRESSIM_SB_STORE(g_SoftEdgeLambdas, edgeIndex, 0.0);
        }
    }

    CRESSIM_SB_STORE(g_SoftEdges, edgeIndex, edge);
}
