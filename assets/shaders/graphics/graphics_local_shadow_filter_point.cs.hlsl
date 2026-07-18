#include "graphics_local_shadow_common.hlsli"

struct IndirectCommandDesc
{
    uint visibleOffset;
    uint maxVisibleCount;
    uint indexCount;
    uint reserved;
};

cbuffer GraphicsLocalShadowPrepareConstants
{
    uint g_EnvCount;
    uint g_MaxObjectsPerEnv;
    uint g_MaxLightsPerEnv;
    uint g_LocalShadowBucketCount;
};

CRESSIM_STRUCTURED_BUFFER(IndirectCommandDesc, g_CommandDescs);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_CommandCountsRW);
CRESSIM_RW_STRUCTURED_BUFFER(VisiblePairInstance, g_VisiblePairsRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint totalFaceCount = localShadowTotalPointFaceCount(g_EnvCount);
    const uint totalObjectCount = localShadowTotalObjectCount(g_EnvCount, g_MaxObjectsPerEnv);
    const uint globalIndex = dispatchThreadId.x;
    if (totalFaceCount == 0u || totalObjectCount == 0u ||
        globalIndex >= totalFaceCount * totalObjectCount)
    {
        return;
    }

    const uint faceIndex = globalIndex / totalObjectCount;
    const uint objectIndex = globalIndex % totalObjectCount;
    const uint envIndex = objectIndex / g_MaxObjectsPerEnv;
    if (envIndex != localShadowPointFaceIndexToEnv(faceIndex))
    {
        return;
    }

    const RenderableQueueInfo queueInfo = CRESSIM_SB_LOAD(g_RenderableQueueInfo, objectIndex);
    if (queueInfo.localShadowCommandIndex == 0xffffffffu)
    {
        return;
    }

    const uint pointSlot = localShadowPointFaceIndexToPointSlot(faceIndex);
    const uint localFaceIndex = localShadowPointFaceIndexToLocalFace(faceIndex);
    const uint shadowViewIndex = localShadowPointViewIndex(envIndex, pointSlot);
    const LocalShadowView shadowView = CRESSIM_SB_LOAD(g_LocalShadowViews, shadowViewIndex);
    if (shadowView.active == 0u || localFaceIndex >= shadowView.layerCount)
    {
        return;
    }

    bool valid = false;
    float3 corners[8];
    localShadowBuildRenderableCorners(objectIndex, valid, corners);
    if (!valid ||
        !localShadowMatrixIntersectsCorners(corners,
                                            shadowView.lightViewProjectionMatrices[localFaceIndex]))
    {
        return;
    }

    const uint commandIndex = queueInfo.localShadowCommandIndex * totalFaceCount + faceIndex;
    const IndirectCommandDesc desc = CRESSIM_SB_LOAD(g_CommandDescs, commandIndex);
    uint visibleSlot = 0u;
    InterlockedAdd(CRESSIM_SB_REF(g_CommandCountsRW, commandIndex), 1u, visibleSlot);
    if (visibleSlot < desc.maxVisibleCount)
    {
        VisiblePairInstance pair;
        pair.objectIndex = objectIndex;
        pair.batchCameraIndex = shadowViewIndex;
        pair.bucketIndex = commandIndex;
        pair.shadowSubviewIndex = localFaceIndex;
        CRESSIM_SB_STORE(g_VisiblePairsRW, desc.visibleOffset + visibleSlot, pair);
    }
}
