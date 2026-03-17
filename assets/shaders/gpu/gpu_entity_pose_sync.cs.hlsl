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

StructuredBuffer<float4> g_SourcePositions;
StructuredBuffer<float4> g_SourceOrientations;
StructuredBuffer<float4> g_SourceScales;
StructuredBuffer<PoseMappingEntry> g_Mappings;

RWStructuredBuffer<float4> g_EntityPositions;
RWStructuredBuffer<float4> g_EntityOrientations;
RWStructuredBuffer<float4> g_EntityScales;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint mappingIndex = dispatchThreadId.x;
    if (mappingIndex >= g_MappingCount)
    {
        return;
    }

    const PoseMappingEntry mapping = g_Mappings[mappingIndex];
    g_EntityPositions[mapping.entityPoseIndex] = g_SourcePositions[mapping.sourcePoseIndex];
    g_EntityOrientations[mapping.entityPoseIndex] = g_SourceOrientations[mapping.sourcePoseIndex];
    g_EntityScales[mapping.entityPoseIndex] = g_SourceScales[mapping.sourcePoseIndex];
}
