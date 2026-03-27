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
    uint g_BatchCameraCount;
    uint g_QueueMode;
    uint g_FilterPadding0;
};

StructuredBuffer<IndirectCommandDesc> g_CommandDescs;
RWStructuredBuffer<uint> g_CommandCountsRW;
RWStructuredBuffer<VisiblePairInstance> g_VisiblePairsRW;

static const uint kQueueModeOpaque = 0u;
static const uint kQueueModeShadow = 1u;
static const uint kInvalidCommandIndex = 0xffffffffu;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalIndex = dispatchThreadId.x;
    if (g_ObjectCount == 0u || g_BatchCameraCount == 0u)
    {
        return;
    }

    const uint totalCount = g_ObjectCount * g_BatchCameraCount;
    if (globalIndex >= totalCount)
    {
        return;
    }

    const uint batchCameraIndex = globalIndex / g_ObjectCount;
    const uint localObjectIndex = globalIndex % g_ObjectCount;
    const BatchCameraMetadata batchCamera = g_BatchCameras[batchCameraIndex];
    const uint cameraIndex = batchCamera.globalCameraIndex;
    const PreparedCamera preparedCamera = g_PreparedCameras[cameraIndex];
    if (preparedCamera.active == 0u || localObjectIndex >= preparedCamera.objectRangeCount)
    {
        return;
    }

    const uint globalObjectIndex = preparedCamera.objectRangeStart + localObjectIndex;
    const uint visibilityIndex = preparedCamera.visibilityDataOffset + localObjectIndex;
    const RenderableQueueInfo queueInfo = g_RenderableQueueInfo[globalObjectIndex];

    if (g_QueueMode == kQueueModeOpaque)
    {
        if (queueInfo.opaqueCommandIndex == kInvalidCommandIndex ||
            CRESSIM_SB_LOAD(g_RenderableVisibilityFlags, visibilityIndex) == 0u)
        {
            return;
        }

        const uint commandIndex = queueInfo.opaqueCommandIndex;
        const IndirectCommandDesc desc = g_CommandDescs[commandIndex];
        uint visibleSlot = 0u;
        InterlockedAdd(g_CommandCountsRW[commandIndex], 1u, visibleSlot);
        if (visibleSlot < desc.maxVisibleCount)
        {
            VisiblePairInstance pair;
            pair.objectIndex = globalObjectIndex;
            pair.batchCameraIndex = batchCameraIndex;
            pair.bucketIndex = commandIndex;
            g_VisiblePairsRW[desc.visibleOffset + visibleSlot] = pair;
        }
        return;
    }

    if (queueInfo.shadowCommandBaseIndex == kInvalidCommandIndex)
    {
        return;
    }

    const uint shadowMask = CRESSIM_SB_LOAD(g_RenderableShadowCascadeMasks, visibilityIndex);
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
            VisiblePairInstance pair;
            pair.objectIndex = globalObjectIndex;
            pair.batchCameraIndex = batchCameraIndex;
            pair.bucketIndex = commandIndex;
            g_VisiblePairsRW[desc.visibleOffset + visibleSlot] = pair;
        }
    }
}
