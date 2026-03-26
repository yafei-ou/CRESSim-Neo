struct IndirectCommandDesc
{
    uint visibleOffset;
    uint maxVisibleCount;
    uint indexCount;
    uint reserved;
};

struct DrawIndexedCommand
{
    uint numIndices;
    uint numInstances;
    uint firstIndexLocation;
    int baseVertex;
    uint firstInstanceLocation;
};

cbuffer GraphicsIndirectResetConstants
{
    uint g_CommandCount;
    uint g_CurrentCameraIndex;
    uint g_RenderableCount;
    uint g_ResetPadding0;
};

StructuredBuffer<IndirectCommandDesc> g_CommandDescs;
RWStructuredBuffer<uint> g_CommandCountsRW;
RWStructuredBuffer<DrawIndexedCommand> g_DrawIndexedCommandsRW;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint commandIndex = dispatchThreadId.x;
    if (commandIndex >= g_CommandCount)
    {
        return;
    }

    const IndirectCommandDesc desc = g_CommandDescs[commandIndex];
    g_CommandCountsRW[commandIndex] = 0u;

    DrawIndexedCommand command;
    command.numIndices = desc.indexCount;
    command.numInstances = 0u;
    command.firstIndexLocation = 0u;
    command.baseVertex = 0;
    command.firstInstanceLocation = desc.visibleOffset;
    g_DrawIndexedCommandsRW[commandIndex] = command;
}
