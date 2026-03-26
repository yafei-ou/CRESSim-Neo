#include "graphics/include/graphics_local_shadow_common.hlsli"

cbuffer GraphicsLocalShadowPrepareConstants
{
    uint g_EnvCount;
    uint g_MaxObjectsPerEnv;
    uint g_MaxLightsPerEnv;
    uint g_LocalShadowBucketCount;
};

RWStructuredBuffer<uint> g_LocalShadowEnvBoundsRW;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint objectIndex = dispatchThreadId.x;
    const uint totalObjectCount = localShadowTotalObjectCount(g_EnvCount, g_MaxObjectsPerEnv);
    if (objectIndex >= totalObjectCount)
    {
        return;
    }

    const RenderableMetadata metadata = g_RenderableMetadata[objectIndex];
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_ACTIVE) == 0u)
    {
        return;
    }

    const uint envIndex = objectIndex / g_MaxObjectsPerEnv;
    const uint baseIndex = envIndex * CRESSIM_LOCAL_SHADOW_ENV_BOUNDS_WORDS;
    const float3 position = g_EntityPositions[objectIndex].xyz;
    InterlockedMin(g_LocalShadowEnvBoundsRW[baseIndex + 0u],
                   localShadowFloatToOrderedUint(position.x));
    InterlockedMin(g_LocalShadowEnvBoundsRW[baseIndex + 1u],
                   localShadowFloatToOrderedUint(position.y));
    InterlockedMin(g_LocalShadowEnvBoundsRW[baseIndex + 2u],
                   localShadowFloatToOrderedUint(position.z));
    InterlockedMax(g_LocalShadowEnvBoundsRW[baseIndex + 3u],
                   localShadowFloatToOrderedUint(position.x));
    InterlockedMax(g_LocalShadowEnvBoundsRW[baseIndex + 4u],
                   localShadowFloatToOrderedUint(position.y));
    InterlockedMax(g_LocalShadowEnvBoundsRW[baseIndex + 5u],
                   localShadowFloatToOrderedUint(position.z));
    InterlockedOr(g_LocalShadowEnvBoundsRW[baseIndex + 6u], 1u);
}
