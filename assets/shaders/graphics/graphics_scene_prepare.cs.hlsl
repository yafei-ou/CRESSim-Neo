#include "graphics/include/graphics_scene_buffers.hlsli"

cbuffer GraphicsScenePrepareConstants
{
    uint g_CameraCount;
    uint g_MaxObjectsPerEnv;
    uint g_PreparePadding0;
    uint g_PreparePadding1;
};

CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RenderableVisibilityFlagsRW);
CRESSIM_RW_STRUCTURED_BUFFER(uint, g_RenderableShadowCascadeMasksRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint globalIndex = dispatchThreadId.x;
    const uint totalCount = g_MaxObjectsPerEnv * g_CameraCount;
    if (globalIndex >= totalCount || g_MaxObjectsPerEnv == 0u)
    {
        return;
    }

    const uint currentCameraIndex = globalIndex / g_MaxObjectsPerEnv;
    const uint localObjectIndex = globalIndex % g_MaxObjectsPerEnv;
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, currentCameraIndex);
    const uint outputIndex = preparedCamera.visibilityDataOffset + localObjectIndex;
    if (preparedCamera.active == 0u)
    {
        CRESSIM_SB_STORE(g_RenderableVisibilityFlagsRW, outputIndex, 0u);
        CRESSIM_SB_STORE(g_RenderableShadowCascadeMasksRW, outputIndex, 0u);
        return;
    }

    const uint renderableIndex = preparedCamera.objectRangeStart + localObjectIndex;
    const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, renderableIndex);
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_ACTIVE) == 0u)
    {
        CRESSIM_SB_STORE(g_RenderableVisibilityFlagsRW, outputIndex, 0u);
        CRESSIM_SB_STORE(g_RenderableShadowCascadeMasksRW, outputIndex, 0u);
        return;
    }

    const float3 position = CRESSIM_SB_REF(g_EntityPositions, renderableIndex).xyz;
    const float4 orientation = normalize(CRESSIM_SB_LOAD(g_EntityOrientations, renderableIndex));
    const float3 scale = CRESSIM_SB_REF(g_EntityScales, renderableIndex).xyz;

    float3 corners[8];
    corners[0] = metadata.localBoundsMin.xyz;
    corners[1] = float3(metadata.localBoundsMax.x, metadata.localBoundsMin.y, metadata.localBoundsMin.z);
    corners[2] = float3(metadata.localBoundsMax.x, metadata.localBoundsMax.y, metadata.localBoundsMin.z);
    corners[3] = float3(metadata.localBoundsMin.x, metadata.localBoundsMax.y, metadata.localBoundsMin.z);
    corners[4] = float3(metadata.localBoundsMin.x, metadata.localBoundsMin.y, metadata.localBoundsMax.z);
    corners[5] = float3(metadata.localBoundsMax.x, metadata.localBoundsMin.y, metadata.localBoundsMax.z);
    corners[6] = metadata.localBoundsMax.xyz;
    corners[7] = float3(metadata.localBoundsMin.x, metadata.localBoundsMax.y, metadata.localBoundsMax.z);

    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        corners[i] = quaternionRotateVector(orientation, corners[i] * scale) + position;
    }

    bool allLeft = true;
    bool allRight = true;
    bool allBottom = true;
    bool allTop = true;
    bool allNear = true;
    bool allFar = true;
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        const float4 clip = mul(float4(corners[i], 1.0), preparedCamera.viewProjectionMatrix);
        allLeft = allLeft && (clip.x < -clip.w);
        allRight = allRight && (clip.x > clip.w);
        allBottom = allBottom && (clip.y < -clip.w);
        allTop = allTop && (clip.y > clip.w);
        allNear = allNear && (clip.z < 0.0);
        allFar = allFar && (clip.z > clip.w);
    }

    const bool visible = !(allLeft || allRight || allBottom || allTop || allNear || allFar);
    const uint visibilityFlag = visible ? 1u : 0u;
    CRESSIM_SB_STORE(g_RenderableVisibilityFlagsRW, outputIndex, visibilityFlag);

    uint shadowMask = 0u;
    const uint shadowCascadeCount = (uint)round(preparedCamera.mainShadowCascadeCount);
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_SHADOW_CASTER) != 0u && shadowCascadeCount > 0u)
    {
        [loop]
        for (uint cascadeIndex = 0u; cascadeIndex < min(shadowCascadeCount, 4u); ++cascadeIndex)
        {
            bool cascadeAllLeft = true;
            bool cascadeAllRight = true;
            bool cascadeAllBottom = true;
            bool cascadeAllTop = true;
            bool cascadeAllNear = true;
            bool cascadeAllFar = true;
            [unroll]
            for (int i = 0; i < 8; ++i)
            {
                const float4 clip =
                    mul(float4(corners[i], 1.0), preparedCamera.lightViewProjectionMatrices[cascadeIndex]);
                cascadeAllLeft = cascadeAllLeft && (clip.x < -clip.w);
                cascadeAllRight = cascadeAllRight && (clip.x > clip.w);
                cascadeAllBottom = cascadeAllBottom && (clip.y < -clip.w);
                cascadeAllTop = cascadeAllTop && (clip.y > clip.w);
                cascadeAllNear = cascadeAllNear && (clip.z < 0.0);
                cascadeAllFar = cascadeAllFar && (clip.z > clip.w);
            }

            if (!(cascadeAllLeft || cascadeAllRight || cascadeAllBottom || cascadeAllTop ||
                  cascadeAllNear || cascadeAllFar))
            {
                shadowMask |= (1u << cascadeIndex);
            }
        }
    }

    CRESSIM_SB_STORE(g_RenderableShadowCascadeMasksRW, outputIndex, shadowMask);
}
