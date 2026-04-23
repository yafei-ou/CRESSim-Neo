#include "include/graphics/graphics_local_shadow_common.hlsli"

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
    const uint objectIndex = dispatchThreadId.x;
    const uint totalObjectCount = localShadowTotalObjectCount(g_EnvCount, g_MaxObjectsPerEnv);
    if (objectIndex >= totalObjectCount)
    {
        return;
    }

    const RenderableMetadata metadata = CRESSIM_SB_LOAD(g_RenderableMetadata, objectIndex);
    if ((metadata.flags & CRESSIM_RENDERABLE_FLAG_ACTIVE) == 0u)
    {
        return;
    }

    const uint envIndex = objectIndex / g_MaxObjectsPerEnv;
    const uint baseIndex = envIndex * CRESSIM_LOCAL_SHADOW_ENV_BOUNDS_WORDS;
    float3 boundsMin = 0.0;
    float3 boundsMax = 0.0;
    if (metadata.softBodyIndex != CRESSIM_INVALID_SOFT_BODY_INDEX)
    {
        const SoftBodyWorldAabb worldAabb =
            CRESSIM_SB_LOAD(g_SoftBodyWorldAabbs, metadata.softBodyIndex);
        boundsMin = worldAabb.minBounds.xyz;
        boundsMax = worldAabb.maxBounds.xyz;
    }
    else
    {
        const float3 position = CRESSIM_SB_REF(g_EntityPositions, objectIndex).xyz;
        boundsMin = position;
        boundsMax = position;
    }
    InterlockedMin(CRESSIM_SB_REF(g_LocalShadowEnvBoundsRW, baseIndex + 0u),
                   localShadowFloatToOrderedUint(boundsMin.x));
    InterlockedMin(CRESSIM_SB_REF(g_LocalShadowEnvBoundsRW, baseIndex + 1u),
                   localShadowFloatToOrderedUint(boundsMin.y));
    InterlockedMin(CRESSIM_SB_REF(g_LocalShadowEnvBoundsRW, baseIndex + 2u),
                   localShadowFloatToOrderedUint(boundsMin.z));
    InterlockedMax(CRESSIM_SB_REF(g_LocalShadowEnvBoundsRW, baseIndex + 3u),
                   localShadowFloatToOrderedUint(boundsMax.x));
    InterlockedMax(CRESSIM_SB_REF(g_LocalShadowEnvBoundsRW, baseIndex + 4u),
                   localShadowFloatToOrderedUint(boundsMax.y));
    InterlockedMax(CRESSIM_SB_REF(g_LocalShadowEnvBoundsRW, baseIndex + 5u),
                   localShadowFloatToOrderedUint(boundsMax.z));
    InterlockedOr(CRESSIM_SB_REF(g_LocalShadowEnvBoundsRW, baseIndex + 6u), 1u);
}
