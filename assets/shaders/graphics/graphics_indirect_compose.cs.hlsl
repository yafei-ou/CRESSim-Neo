struct DrawIndexedCommand
{
    uint numIndices;
    uint numInstances;
    uint firstIndexLocation;
    int baseVertex;
    uint firstInstanceLocation;
};

cbuffer GraphicsIndirectComposeConstants
{
    uint g_CommandCount;
    uint g_CurrentCameraIndex;
    uint g_RenderableCount;
    uint g_ComposePadding0;
};

StructuredBuffer<uint> g_CommandCounts;
RWStructuredBuffer<DrawIndexedCommand> g_DrawIndexedCommandsRW;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint commandIndex = dispatchThreadId.x;
    if (commandIndex >= g_CommandCount)
    {
        return;
    }

    DrawIndexedCommand command = g_DrawIndexedCommandsRW[commandIndex];
    command.numInstances = g_CommandCounts[commandIndex];
    g_DrawIndexedCommandsRW[commandIndex] = command;
}
