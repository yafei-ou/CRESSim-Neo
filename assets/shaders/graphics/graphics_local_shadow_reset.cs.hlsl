#include "graphics/include/graphics_local_shadow_common.hlsli"

cbuffer GraphicsLocalShadowPrepareConstants
{
    uint g_EnvCount;
    uint g_MaxObjectsPerEnv;
    uint g_MaxLightsPerEnv;
    uint g_LocalShadowBucketCount;
};

RWStructuredBuffer<LightShadowAssignment> g_LightShadowAssignmentsRW;
RWStructuredBuffer<LocalShadowView> g_LocalShadowViewsRW;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    const uint totalLightCount = localShadowTotalLightCount(g_EnvCount, g_MaxLightsPerEnv);
    const uint totalViewCount = localShadowTotalViewCount(g_EnvCount);

    if (index < totalLightCount)
    {
        LightShadowAssignment assignment = (LightShadowAssignment)0;
        assignment.shadowViewIndex = CRESSIM_INVALID_GPU_SCENE_INDEX;
        g_LightShadowAssignmentsRW[index] = assignment;
    }

    if (index < totalViewCount)
    {
        LocalShadowView shadowView = (LocalShadowView)0;
        shadowView.lightIndex = CRESSIM_INVALID_GPU_SCENE_INDEX;
        shadowView.active = 0u;
        g_LocalShadowViewsRW[index] = shadowView;
    }
}
