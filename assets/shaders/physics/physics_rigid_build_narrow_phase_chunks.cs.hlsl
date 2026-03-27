#include "physics/include/physics_rigid_common.hlsli"

static const uint kNarrowPhaseChunkSize = 128u;

StructuredBuffer<GpuRigidPairRange> g_RigidPairRanges;
RWStructuredBuffer<GpuNarrowPhaseChunk> g_NarrowPhaseChunks;
RWStructuredBuffer<GpuNarrowPhaseMeta> g_NarrowPhaseMeta;
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_NarrowPhaseChunkCounter);

[numthreads(1, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u)
    {
        return;
    }

    uint chunkIndex = 0u;
    [unroll] for (uint type = 0u; type < kRigidPairTypeCount; ++type)
    {
        const GpuRigidPairRange range = g_RigidPairRanges[type];
        uint pairStart = range.start;
        uint remaining = range.count;
        while (remaining > 0u)
        {
            GpuNarrowPhaseChunk chunk;
            chunk.pairType = range.type;
            chunk.pairStart = pairStart;
            chunk.pairCount = min(remaining, kNarrowPhaseChunkSize);
            chunk.reserved = 0u;
            g_NarrowPhaseChunks[chunkIndex++] = chunk;
            pairStart += chunk.pairCount;
            remaining -= chunk.pairCount;
        }
    }

    GpuNarrowPhaseMeta meta;
    meta.chunkCount = chunkIndex;
    meta.reserved0 = 0u;
    meta.reserved1 = 0u;
    meta.reserved2 = 0u;
    g_NarrowPhaseMeta[0] = meta;
    CRESSIM_SB_STORE(g_NarrowPhaseChunkCounter, 0u, 0u);
}
