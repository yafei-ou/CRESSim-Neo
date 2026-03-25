#include "graphics/include/graphics_shadow_constants.hlsli"
#include "graphics/include/graphics_scene_buffers.hlsli"

struct VSInput
{
    float3 Position : ATTRIB0;
    float3 Normal : ATTRIB1;
    float2 TexCoord : ATTRIB2;
};

struct VSOutput
{
    float4 Position : SV_Position;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
};

void main(in VSInput In, out VSOutput Out, uint instanceId : SV_InstanceID)
{
    uint objectIndex = g_InstanceIndex;
    uint cameraIndex = 0u;
    uint shadowLayer = 0u;
    const bool localShadowPass = g_ShadowPassMode != 0u;
    if (g_UseDrawListBuffer != 0u)
    {
        const VisiblePairInstance pair = g_VisiblePairs[g_DrawListOffset + instanceId];
        objectIndex = pair.objectIndex;
        if (localShadowPass)
        {
            const LocalShadowView shadowView = g_LocalShadowViews[pair.batchCameraIndex];
            shadowLayer = g_ShadowMatrixIndex;

            bool poseValid = false;
            float3 position = float3(0.0, 0.0, 0.0);
            float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
            float3 scale = float3(1.0, 1.0, 1.0);
            loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
            if (!poseValid || shadowView.active == 0u || g_ShadowMatrixIndex >= shadowView.layerCount)
            {
                Out.Position = float4(2.0, 2.0, 2.0, 1.0);
#if MANUAL_LAYER_EXPORT
                Out.Layer = 0u;
#endif
                return;
            }

            const float4 worldPos =
                float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
            Out.Position = mul(worldPos, shadowView.lightViewProjectionMatrices[g_ShadowMatrixIndex]);
#if MANUAL_LAYER_EXPORT
            Out.Layer = g_ShadowMatrixIndex;
#endif
            return;
        }

        const BatchCameraMetadata batchCamera = g_BatchCameras[pair.batchCameraIndex];
        cameraIndex = batchCamera.globalCameraIndex;
        shadowLayer = batchCamera.shadowLayer;
    }
    const PreparedCamera preparedCamera = g_PreparedCameras[cameraIndex];
    bool poseValid = false;
    float3 position = float3(0.0, 0.0, 0.0);
    float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
    float3 scale = float3(1.0, 1.0, 1.0);
    loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
    const uint localObjectIndex = objectIndex - preparedCamera.objectRangeStart;
    const uint shadowMask =
        g_RenderableShadowCascadeMasks[preparedCamera.visibilityDataOffset + localObjectIndex];
    if (!poseValid || preparedCamera.active == 0u ||
        shadowLayer == CRESSIM_INVALID_BATCH_CAMERA_LAYER ||
        ((shadowMask & (1u << g_CascadeIndex)) == 0u))
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = shadowLayer;
#endif
        return;
    }
    const float4 worldPos =
        float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
    Out.Position = mul(worldPos, preparedCamera.lightViewProjectionMatrices[g_CascadeIndex]);
#if MANUAL_LAYER_EXPORT
    Out.Layer = shadowLayer;
#endif
}
