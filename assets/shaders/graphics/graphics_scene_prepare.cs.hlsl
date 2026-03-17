#include "graphics/include/graphics_scene_buffers.hlsli"

cbuffer GraphicsScenePrepareConstants
{
    float4x4 g_ViewProjection;
    float4x4 g_LightViewProjection[4];
    uint g_RenderableCount;
    uint g_ShadowCascadeCount;
    uint g_CameraEnvIndex;
    uint g_PreparePadding;
};

RWStructuredBuffer<uint> g_RenderableVisibilityFlagsRW;
RWStructuredBuffer<uint> g_RenderableShadowCascadeMasksRW;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint renderableIndex = dispatchThreadId.x;
    if (renderableIndex >= g_RenderableCount)
    {
        return;
    }

    const RenderableMetadata metadata = g_RenderableMetadata[renderableIndex];
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_GPU_POSE) == 0u ||
        metadata.entityPoseIndex == 0xffffffffu || metadata.envIndex != g_CameraEnvIndex)
    {
        g_RenderableVisibilityFlagsRW[renderableIndex] = 0u;
        g_RenderableShadowCascadeMasksRW[renderableIndex] = 0u;
        return;
    }

    const float3 position = g_EntityPositions[metadata.entityPoseIndex].xyz;
    const float4 orientation = normalize(g_EntityOrientations[metadata.entityPoseIndex]);
    const float3 scale = g_EntityScales[metadata.entityPoseIndex].xyz;

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
        const float4 clip = mul(float4(corners[i], 1.0), g_ViewProjection);
        allLeft = allLeft && (clip.x < -clip.w);
        allRight = allRight && (clip.x > clip.w);
        allBottom = allBottom && (clip.y < -clip.w);
        allTop = allTop && (clip.y > clip.w);
        allNear = allNear && (clip.z < 0.0);
        allFar = allFar && (clip.z > clip.w);
    }

    const bool visible = !(allLeft || allRight || allBottom || allTop || allNear || allFar);
    g_RenderableVisibilityFlagsRW[renderableIndex] = visible ? 1u : 0u;

    uint shadowMask = 0u;
    if ((metadata.flags & (1u << 1u)) != 0u)
    {
        [loop]
        for (uint cascadeIndex = 0u; cascadeIndex < min(g_ShadowCascadeCount, 4u); ++cascadeIndex)
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
                const float4 clip = mul(float4(corners[i], 1.0), g_LightViewProjection[cascadeIndex]);
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

    g_RenderableShadowCascadeMasksRW[renderableIndex] = shadowMask;
}
