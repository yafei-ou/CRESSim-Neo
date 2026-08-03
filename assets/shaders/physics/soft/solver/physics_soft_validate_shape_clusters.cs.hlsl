#include "physics_particle_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(GpuShapeCluster, g_ShapeClusters);
CRESSIM_STRUCTURED_BUFFER(GpuShapeClusterLink, g_ShapeClusterLinks);
CRESSIM_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);

[numthreads(128, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint clusterIndex = dispatchThreadID.x;
    if (clusterIndex >= shapeClusterCount)
    {
        return;
    }

    GpuShapeCluster cluster = CRESSIM_SB_LOAD(g_ShapeClusters, clusterIndex);
    if ((cluster.flags & kShapeClusterActive) == 0u ||
        (cluster.flags & kShapeClusterCutDisabled) != 0u ||
        cluster.memberCount == 0u ||
        cluster.memberCount > 16u)
    {
        return;
    }

    uint adjacency[16];
    [unroll]
    for (uint i = 0u; i < 16u; ++i)
    {
        adjacency[i] = 0u;
    }

    for (uint linkIndex = 0u; linkIndex < cluster.linkCount; ++linkIndex)
    {
        const GpuShapeClusterLink link =
            CRESSIM_SB_LOAD(g_ShapeClusterLinks, cluster.linkOffset + linkIndex);
        if (link.localParticleA >= cluster.memberCount ||
            link.localParticleB >= cluster.memberCount)
        {
            continue;
        }

        const GpuSoftEdge edge = CRESSIM_SB_LOAD(g_SoftEdges, link.softEdgeIndex);
        if ((edge.flags & kSoftEdgeActiveFlag) == 0u ||
            (edge.flags & kSoftEdgeDisabledFlag) != 0u)
        {
            continue;
        }

        adjacency[link.localParticleA] |= 1u << link.localParticleB;
        adjacency[link.localParticleB] |= 1u << link.localParticleA;
    }

    uint visited = 1u;
    for (uint iteration = 0u; iteration < cluster.memberCount; ++iteration)
    {
        uint expanded = visited;
        for (uint localIndex = 0u; localIndex < cluster.memberCount; ++localIndex)
        {
            if ((visited & (1u << localIndex)) != 0u)
            {
                expanded |= adjacency[localIndex];
            }
        }

        if (expanded == visited)
        {
            break;
        }
        visited = expanded;
    }

    const uint requiredMask = (1u << cluster.memberCount) - 1u;
    if ((visited & requiredMask) != requiredMask)
    {
        cluster.flags &= ~kShapeClusterActive;
        cluster.flags |= kShapeClusterCutDisabled;
        CRESSIM_SB_STORE(g_ShapeClusters, clusterIndex, cluster);
    }
}
