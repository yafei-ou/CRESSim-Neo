cbuffer GpuEntityPoseSyncConstantsBuffer
{
    uint g_MappingCount;
    uint g_Padding0;
    uint g_Padding1;
    uint g_Padding2;
};

struct PoseMappingEntry
{
    uint sourcePoseIndex;
    uint entityPoseIndex;
};

#include "include/structured_buffer_compat.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_SourcePositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_SourceOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_SourceScales);
CRESSIM_STRUCTURED_BUFFER(PoseMappingEntry, g_Mappings);

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityPositions);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityOrientations);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_EntityScales);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint mappingIndex = dispatchThreadId.x;
    if (mappingIndex >= g_MappingCount)
    {
        return;
    }

    const PoseMappingEntry mapping = CRESSIM_SB_LOAD(g_Mappings, mappingIndex);
    CRESSIM_SB_STORE(g_EntityPositions, mapping.entityPoseIndex,
                     CRESSIM_SB_LOAD(g_SourcePositions, mapping.sourcePoseIndex));
    CRESSIM_SB_STORE(g_EntityOrientations, mapping.entityPoseIndex,
                     CRESSIM_SB_LOAD(g_SourceOrientations, mapping.sourcePoseIndex));
    CRESSIM_SB_STORE(g_EntityScales, mapping.entityPoseIndex,
                     CRESSIM_SB_LOAD(g_SourceScales, mapping.sourcePoseIndex));
}
