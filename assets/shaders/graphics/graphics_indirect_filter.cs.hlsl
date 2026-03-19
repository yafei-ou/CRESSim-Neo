#include "graphics/include/graphics_scene_buffers.hlsli"

struct IndirectCommandDesc
{
    uint visibleOffset;
    uint maxVisibleCount;
    uint indexCount;
    uint reserved;
};

cbuffer GraphicsIndirectFilterConstants
{
    uint g_ObjectCount;
    uint g_CurrentCameraIndex;
    uint g_QueueMode;
    uint g_FilterPadding0;
};

StructuredBuffer<IndirectCommandDesc> g_CommandDescs;
RWStructuredBuffer<uint> g_CommandCountsRW;
RWStructuredBuffer<uint> g_VisibleObjectIndicesRW;

static const uint kQueueModeOpaque = 0u;
static const uint kQueueModeShadow = 1u;
static const uint kInvalidCommandIndex = 0xffffffffu;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const PreparedCamera preparedCamera = g_PreparedCameras[g_CurrentCameraIndex];
    const uint localObjectIndex = dispatchThreadId.x;
    if (localObjectIndex >= preparedCamera.objectRangeCount || localObjectIndex >= g_ObjectCount)
    {
        return;
    }

    const uint globalObjectIndex = preparedCamera.objectRangeStart + localObjectIndex;
    const uint visibilityIndex = preparedCamera.visibilityDataOffset + localObjectIndex;
    const RenderableQueueInfo queueInfo = g_RenderableQueueInfo[globalObjectIndex];

    if (g_QueueMode == kQueueModeOpaque)
    {
        if (queueInfo.opaqueCommandIndex == kInvalidCommandIndex ||
            g_RenderableVisibilityFlags[visibilityIndex] == 0u)
        {
            return;
        }

        const IndirectCommandDesc desc = g_CommandDescs[queueInfo.opaqueCommandIndex];
        uint visibleSlot = 0u;
        InterlockedAdd(g_CommandCountsRW[queueInfo.opaqueCommandIndex], 1u, visibleSlot);
        if (visibleSlot < desc.maxVisibleCount)
        {
            g_VisibleObjectIndicesRW[desc.visibleOffset + visibleSlot] = globalObjectIndex;
        }
        return;
    }

    if (queueInfo.shadowCommandBaseIndex == kInvalidCommandIndex)
    {
        return;
    }

    const uint shadowMask = g_RenderableShadowCascadeMasks[visibilityIndex];
    [unroll]
    for (uint cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex)
    {
        if ((shadowMask & (1u << cascadeIndex)) == 0u)
        {
            continue;
        }

        const uint commandIndex = queueInfo.shadowCommandBaseIndex + cascadeIndex;
        const IndirectCommandDesc desc = g_CommandDescs[commandIndex];
        uint visibleSlot = 0u;
        InterlockedAdd(g_CommandCountsRW[commandIndex], 1u, visibleSlot);
        if (visibleSlot < desc.maxVisibleCount)
        {
            g_VisibleObjectIndicesRW[desc.visibleOffset + visibleSlot] = globalObjectIndex;
        }
    }
}
