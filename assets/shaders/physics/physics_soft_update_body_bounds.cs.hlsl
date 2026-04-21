#include "include/physics/physics_soft_render_dispatch_constants.hlsli"
#include "include/physics/physics_rigid_common.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuSoftBodyBoundsChunk, g_SoftBodyBoundsChunks);
CRESSIM_RW_STRUCTURED_BUFFER(GpuBodyAabb, g_SoftBodyChunkAabbsRW);

groupshared float3 s_MinBounds[64];
groupshared float3 s_MaxBounds[64];

[numthreads(64, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint chunkIndex = groupId.x;
    if (chunkIndex >= softRenderReserved0)
    {
        return;
    }

    const GpuSoftBodyBoundsChunk chunk = CRESSIM_SB_LOAD(g_SoftBodyBoundsChunks, chunkIndex);
    const uint lane = groupThreadId.x;
    GpuBodyAabb bodyAabb;
    if (chunk.particleCount == 0u)
    {
        if (lane == 0u)
        {
            bodyAabb.minBounds = float4(0.0, 0.0, 0.0, 0.0);
            bodyAabb.maxBounds = float4(0.0, 0.0, 0.0, 0.0);
            CRESSIM_SB_STORE(g_SoftBodyChunkAabbsRW, chunkIndex, bodyAabb);
        }
        return;
    }

    float3 minBounds = float3(3.402823e+38, 3.402823e+38, 3.402823e+38);
    float3 maxBounds = float3(-3.402823e+38, -3.402823e+38, -3.402823e+38);
    for (uint i = lane; i < chunk.particleCount; i += 64u)
    {
        const float3 position =
            CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, chunk.particleStart + i).xyz;
        minBounds = min(minBounds, position);
        maxBounds = max(maxBounds, position);
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
        bodyAabb.minBounds = float4(s_MinBounds[0], 0.0);
        bodyAabb.maxBounds = float4(s_MaxBounds[0], 0.0);
        CRESSIM_SB_STORE(g_SoftBodyChunkAabbsRW, chunkIndex, bodyAabb);
    }
}
