#include "graphics/graphics_local_shadow_common.hlsli"

cbuffer GraphicsLocalShadowPrepareConstants
{
    uint g_EnvCount;
    uint g_MaxObjectsPerEnv;
    uint g_MaxLightsPerEnv;
    uint g_LocalShadowBucketCount;
};

CRESSIM_RW_STRUCTURED_BUFFER(uint, g_LocalShadowEnvBoundsRW);

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
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 0u, orderedHuge);
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 1u, orderedHuge);
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 2u, orderedHuge);
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 3u, orderedNegHuge);
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 4u, orderedNegHuge);
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 5u, orderedNegHuge);
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 6u, 0u);
    CRESSIM_SB_STORE(g_LocalShadowEnvBoundsRW, baseIndex + 7u, 0u);
}
