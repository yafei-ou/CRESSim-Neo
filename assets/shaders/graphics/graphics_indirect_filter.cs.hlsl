#include "include/graphics/graphics_scene_buffers.hlsli"

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

CRESSIM_STRUCTURED_BUFFER(IndirectCommandDesc, g_CommandDescs);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_CommandCountsRW);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_VisibleObjectIndicesRW);

static const uint kQueueModeOpaque = 0u;
static const uint kQueueModeShadow = 1u;
static const uint kInvalidCommandIndex = 0xffffffffu;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, g_CurrentCameraIndex);
    const uint localObjectIndex = dispatchThreadId.x;
    if (localObjectIndex >= preparedCamera.objectRangeCount || localObjectIndex >= g_ObjectCount)
    {
        return;
    }

    const uint globalObjectIndex = preparedCamera.objectRangeStart + localObjectIndex;
    const uint visibilityIndex = preparedCamera.visibilityDataOffset + localObjectIndex;
    const RenderableQueueInfo queueInfo = CRESSIM_SB_LOAD(g_RenderableQueueInfo, globalObjectIndex);

    if (g_QueueMode == kQueueModeOpaque)
    {
        if (queueInfo.opaqueCommandIndex == kInvalidCommandIndex ||
            CRESSIM_SB_LOAD(g_RenderableVisibilityFlags, visibilityIndex) == 0u)
        {
            return;
        }

        const IndirectCommandDesc desc = CRESSIM_SB_LOAD(g_CommandDescs, queueInfo.opaqueCommandIndex);
        uint visibleSlot = 0u;
        InterlockedAdd(CRESSIM_SB_REF(g_CommandCountsRW, queueInfo.opaqueCommandIndex), 1u,
                       visibleSlot);
        if (visibleSlot < desc.maxVisibleCount)
        {
            CRESSIM_SB_STORE(g_VisibleObjectIndicesRW, desc.visibleOffset + visibleSlot,
                             globalObjectIndex);
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
        const IndirectCommandDesc desc = CRESSIM_SB_LOAD(g_CommandDescs, commandIndex);
        uint visibleSlot = 0u;
        InterlockedAdd(CRESSIM_SB_REF(g_CommandCountsRW, commandIndex), 1u, visibleSlot);
        if (visibleSlot < desc.maxVisibleCount)
        {
            CRESSIM_SB_STORE(g_VisibleObjectIndicesRW, desc.visibleOffset + visibleSlot,
                             globalObjectIndex);
        }
    }
}
