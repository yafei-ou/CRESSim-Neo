#include "include/graphics/graphics_local_shadow_common.hlsli"

cbuffer GraphicsLocalShadowPrepareConstants
{
    uint g_EnvCount;
    uint g_MaxObjectsPerEnv;
    uint g_MaxLightsPerEnv;
    uint g_LocalShadowBucketCount;
};

CRESSIM_STRUCTURED_BUFFER(uint, g_LocalShadowEnvBounds);
CRESSIM_RW_STRUCTURED_BUFFER(LightShadowAssignment, g_LightShadowAssignmentsRW);
CRESSIM_RW_STRUCTURED_BUFFER(LocalShadowView, g_LocalShadowViewsRW);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint envIndex = dispatchThreadId.x;
    if (envIndex >= g_EnvCount)
    {
        return;
    }

    const LocalLightSelection selection = CRESSIM_SB_LOAD(g_LocalLightSelections, envIndex);
    uint local2DSlot = 0u;
    uint pointSlot = 0u;
    bool envBoundsValid = false;
    float3 envBoundsCenter = float3(0.0, 0.0, 0.0);
    float envBoundsRadius = 10.0;
    const uint envBoundsBaseIndex = envIndex * CRESSIM_LOCAL_SHADOW_ENV_BOUNDS_WORDS;
    envBoundsValid = CRESSIM_SB_LOAD(g_LocalShadowEnvBounds, envBoundsBaseIndex + 6u) != 0u;
    if (envBoundsValid)
    {
        const float3 envBoundsMin = float3(
            localShadowOrderedUintToFloat(CRESSIM_SB_LOAD(g_LocalShadowEnvBounds,
                                                          envBoundsBaseIndex + 0u)),
            localShadowOrderedUintToFloat(CRESSIM_SB_LOAD(g_LocalShadowEnvBounds,
                                                          envBoundsBaseIndex + 1u)),
            localShadowOrderedUintToFloat(CRESSIM_SB_LOAD(g_LocalShadowEnvBounds,
                                                          envBoundsBaseIndex + 2u)));
        const float3 envBoundsMax = float3(
            localShadowOrderedUintToFloat(CRESSIM_SB_LOAD(g_LocalShadowEnvBounds,
                                                          envBoundsBaseIndex + 3u)),
            localShadowOrderedUintToFloat(CRESSIM_SB_LOAD(g_LocalShadowEnvBounds,
                                                          envBoundsBaseIndex + 4u)),
            localShadowOrderedUintToFloat(CRESSIM_SB_LOAD(g_LocalShadowEnvBounds,
                                                          envBoundsBaseIndex + 5u)));
        envBoundsCenter = (envBoundsMin + envBoundsMax) * 0.5;
        const float3 envBoundsExtents = envBoundsMax - envBoundsMin;
        envBoundsRadius = max(5.0, length(envBoundsExtents) * 0.5 + 2.0);
    }

    [unroll]
    for (uint selectionIndex = 0u; selectionIndex < CRESSIM_FORWARD_LOCAL_LIGHT_CAP; ++selectionIndex)
    {
        if (selectionIndex >= selection.localLightCount)
        {
            break;
        }

        const uint lightIndex = selection.lightIndices[selectionIndex];
        if (lightIndex >= localShadowTotalLightCount(g_EnvCount, g_MaxLightsPerEnv))
        {
            continue;
        }

        const LightInput light = CRESSIM_SB_LOAD(g_LightInputs, lightIndex);
        const bool active =
            light.active != 0u && light.castsShadows != 0u &&
            light.directionIntensity.w > 0.0 &&
            (light.type != CRESSIM_LIGHT_TYPE_DIRECTIONAL ||
             dot(light.directionIntensity.xyz, light.directionIntensity.xyz) > 1e-6);
        if (!active)
        {
            continue;
        }

        const bool pointLight = light.type == CRESSIM_LIGHT_TYPE_POINT;
        if (pointLight)
        {
            if (pointSlot >= CRESSIM_SHADOWED_POINT_LIGHT_CAP)
            {
                continue;
            }

            const uint shadowViewIndex = localShadowPointViewIndex(envIndex, pointSlot);
            LocalShadowView shadowView = (LocalShadowView)0;
            shadowView.lightIndex = lightIndex;
            shadowView.envIndex = envIndex;
            shadowView.firstLayer = localShadowPointFirstLayer(envIndex, pointSlot);
            shadowView.layerCount = CRESSIM_LOCAL_SHADOW_POINT_FACE_COUNT;
            shadowView.lightType = light.type;
            shadowView.active = 1u;
            shadowView.lightPositionRange = light.positionRange;
            shadowView.lightDirection = float4(0.0, 0.0, 0.0, 0.0);
            shadowView.shadowTexelSize = float2(1.0 / 512.0, 1.0 / 512.0);
            shadowView.shadowNearPlane = 0.05;
            shadowView.shadowFarPlane = max(light.positionRange.w, 1.0);

            const float3 position = light.positionRange.xyz;
            const float3 targets[6] = {
                float3(1.0, 0.0, 0.0), float3(-1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0),
                float3(0.0, -1.0, 0.0), float3(0.0, 0.0, 1.0), float3(0.0, 0.0, -1.0)
            };
            const float3 ups[6] = {
                float3(0.0, 1.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, -1.0),
                float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 0.0), float3(0.0, 1.0, 0.0)
            };
            [unroll]
            for (uint faceIndex = 0u; faceIndex < 6u; ++faceIndex)
            {
                const float4x4 view = localShadowBuildLookAtMatrix(position,
                                                                   position + targets[faceIndex],
                                                                   ups[faceIndex]);
                const float4x4 proj =
                    localShadowBuildPerspectiveMatrix(CRESSIM_LOCAL_SHADOW_POINT_FACE_FOV_RADIANS,
                                                      1.0, 0.05, max(light.positionRange.w, 1.0));
                shadowView.lightViewProjectionMatrices[faceIndex] = mul(view, proj);
            }

            LightShadowAssignment assignment = (LightShadowAssignment)0;
            assignment.shadowMode = 2u;
            assignment.shadowViewIndex = shadowViewIndex;
            CRESSIM_SB_STORE(g_LightShadowAssignmentsRW, lightIndex, assignment);
            CRESSIM_SB_STORE(g_LocalShadowViewsRW, shadowViewIndex, shadowView);
            ++pointSlot;
            continue;
        }

        if (local2DSlot >= CRESSIM_SHADOWED_LOCAL_LIGHT_CAP)
        {
            continue;
        }

        const uint shadowViewIndex = localShadow2DViewIndex(envIndex, local2DSlot);
        const float3 lightDirection =
            localShadowSafeNormalize(light.directionIntensity.xyz, float3(0.0, -1.0, 0.0));

        LocalShadowView shadowView = (LocalShadowView)0;
        shadowView.lightIndex = lightIndex;
        shadowView.envIndex = envIndex;
        shadowView.firstLayer = localShadow2DLayer(envIndex, local2DSlot);
        shadowView.layerCount = 1u;
        shadowView.lightType = light.type;
        shadowView.active = 1u;
        shadowView.lightPositionRange = light.positionRange;
        shadowView.lightDirection = float4(lightDirection, 0.0);
        shadowView.shadowTexelSize = float2(1.0 / 1024.0, 1.0 / 1024.0);

        if (light.type == CRESSIM_LIGHT_TYPE_SPOT)
        {
            shadowView.shadowNearPlane = 0.05;
            shadowView.shadowFarPlane =
                max(light.positionRange.w * CRESSIM_LOCAL_SHADOW_SPOT_RANGE_GUARD_SCALE, 1.0);
            const float4x4 view = localShadowBuildLookAtMatrix(
                light.positionRange.xyz, light.positionRange.xyz + lightDirection, float3(0.0, 1.0, 0.0));
            const float fovRadians =
                max(light.spotAngles.w * 2.0 * (CRESSIM_LOCAL_SHADOW_PI / 180.0) +
                        CRESSIM_LOCAL_SHADOW_SPOT_FOV_GUARD_RADIANS,
                                         CRESSIM_LOCAL_SHADOW_PI / 180.0);
            const float4x4 proj =
                localShadowBuildPerspectiveMatrix(fovRadians, 1.0, 0.05,
                                                  shadowView.shadowFarPlane);
            shadowView.lightViewProjectionMatrices[0] = mul(view, proj);
        }
        else
        {
            const float radius = envBoundsValid ? envBoundsRadius : max(light.positionRange.w, 10.0);
            const float3 center = envBoundsValid ? envBoundsCenter : float3(0.0, 0.0, 0.0);
            const float3 eye =
                center - lightDirection * (radius * 2.0 + CRESSIM_LOCAL_SHADOW_DIRECTIONAL_DEPTH_PADDING);
            shadowView.shadowNearPlane = 0.1;
            shadowView.shadowFarPlane = radius * 4.0 + CRESSIM_LOCAL_SHADOW_DIRECTIONAL_DEPTH_PADDING * 2.0;
            const float3 upCandidate =
                abs(lightDirection.y) > 0.98 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
            const float4x4 view = localShadowBuildLookAtMatrix(eye, center, upCandidate);
            const float4x4 proj = localShadowBuildOrthoOffCenterMatrix(
                -radius, radius, -radius, radius, 0.1,
                radius * 4.0 + CRESSIM_LOCAL_SHADOW_DIRECTIONAL_DEPTH_PADDING * 2.0);
            shadowView.lightViewProjectionMatrices[0] = mul(view, proj);
        }

        LightShadowAssignment assignment = (LightShadowAssignment)0;
        assignment.shadowMode = 1u;
        assignment.shadowViewIndex = shadowViewIndex;
        CRESSIM_SB_STORE(g_LightShadowAssignmentsRW, lightIndex, assignment);
        CRESSIM_SB_STORE(g_LocalShadowViewsRW, shadowViewIndex, shadowView);
        ++local2DSlot;
    }
}
