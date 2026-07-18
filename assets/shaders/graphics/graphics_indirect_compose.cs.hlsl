#include "structured_buffer_compat.hlsli"

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

CRESSIM_STRUCTURED_BUFFER(uint, g_CommandCounts);
CRESSIM_RW_STRUCTURED_BUFFER(DrawIndexedCommand, g_DrawIndexedCommandsRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint commandIndex = dispatchThreadId.x;
    if (commandIndex >= g_CommandCount)
    {
        return;
    }

    DrawIndexedCommand command = CRESSIM_SB_LOAD(g_DrawIndexedCommandsRW, commandIndex);
    command.numInstances = CRESSIM_SB_LOAD(g_CommandCounts, commandIndex);
    CRESSIM_SB_STORE(g_DrawIndexedCommandsRW, commandIndex, command);
}
