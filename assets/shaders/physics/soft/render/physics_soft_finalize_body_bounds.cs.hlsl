#include "physics_soft_render_dispatch_constants.hlsli"
#include "physics_particle_types.hlsli"
#include "physics_rigid_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(GpuSoftBodyChunkRange, g_SoftBodyChunkRanges);
CRESSIM_STRUCTURED_BUFFER(GpuBodyAabb, g_SoftBodyChunkAabbs);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBodyAabb, g_SoftBodyWorldAabbsRW);

groupshared float3 s_MinBounds[64];
groupshared float3 s_MaxBounds[64];

[numthreads(64, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint softBodyIndex = groupId.x;
    if (softBodyIndex >= softBodyCount)
    {
        return;
    }

    const GpuSoftBodyChunkRange range = CRESSIM_SB_LOAD(g_SoftBodyChunkRanges, softBodyIndex);
    const uint lane = groupThreadId.x;
    float3 minBounds = float3(3.402823e+38, 3.402823e+38, 3.402823e+38);
    float3 maxBounds = float3(-3.402823e+38, -3.402823e+38, -3.402823e+38);

    if (range.count == 0u)
    {
        if (lane == 0u)
        {
            GpuBodyAabb bodyAabb;
            bodyAabb.minBounds = float4(0.0, 0.0, 0.0, 0.0);
            bodyAabb.maxBounds = float4(0.0, 0.0, 0.0, 0.0);
            CRESSIM_SB_STORE(g_SoftBodyWorldAabbsRW, softBodyIndex, bodyAabb);
        }
        return;
    }

    for (uint i = lane; i < range.count; i += 64u)
    {
        const GpuBodyAabb chunkAabb = CRESSIM_SB_LOAD(g_SoftBodyChunkAabbs, range.start + i);
        minBounds = min(minBounds, chunkAabb.minBounds.xyz);
        maxBounds = max(maxBounds, chunkAabb.maxBounds.xyz);
    }

    s_MinBounds[lane] = minBounds;
    s_MaxBounds[lane] = maxBounds;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 32u; stride > 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            s_MinBounds[lane] = min(s_MinBounds[lane], s_MinBounds[lane + stride]);
            s_MaxBounds[lane] = max(s_MaxBounds[lane], s_MaxBounds[lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0u)
    {
        GpuBodyAabb bodyAabb;
        bodyAabb.minBounds = float4(s_MinBounds[0], 0.0);
        bodyAabb.maxBounds = float4(s_MaxBounds[0], 0.0);
        CRESSIM_SB_STORE(g_SoftBodyWorldAabbsRW, softBodyIndex, bodyAabb);
    }
}
