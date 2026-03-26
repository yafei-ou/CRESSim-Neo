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
    const uint envIndex = dispatchThreadId.x;
    if (envIndex >= g_EnvCount)
    {
        return;
    }

    const uint baseIndex = envIndex * CRESSIM_LOCAL_SHADOW_ENV_BOUNDS_WORDS;
    const uint orderedHuge = localShadowFloatToOrderedUint(1.0e9);
    const uint orderedNegHuge = localShadowFloatToOrderedUint(-1.0e9);
    g_LocalShadowEnvBoundsRW[baseIndex + 0u] = orderedHuge;
    g_LocalShadowEnvBoundsRW[baseIndex + 1u] = orderedHuge;
    g_LocalShadowEnvBoundsRW[baseIndex + 2u] = orderedHuge;
    g_LocalShadowEnvBoundsRW[baseIndex + 3u] = orderedNegHuge;
    g_LocalShadowEnvBoundsRW[baseIndex + 4u] = orderedNegHuge;
    g_LocalShadowEnvBoundsRW[baseIndex + 5u] = orderedNegHuge;
    g_LocalShadowEnvBoundsRW[baseIndex + 6u] = 0u;
    g_LocalShadowEnvBoundsRW[baseIndex + 7u] = 0u;
}
