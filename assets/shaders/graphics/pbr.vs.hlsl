#include "graphics/include/graphics_forward_constants.hlsli"
#include "graphics/include/graphics_scene_buffers.hlsli"

struct VSInput
{
    float3 Position : ATTRIB0;
    float3 Normal : ATTRIB1;
    float2 TexCoord : ATTRIB2;
    float4 Tangent : ATTRIB3;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float2 TexCoord : TEXCOORD2;
    float4 WorldTangent : TEXCOORD3;
    nointerpolation uint CameraIndex : TEXCOORD4;
    nointerpolation uint MainLightIndex : TEXCOORD5;
    nointerpolation uint ShadowLayer : TEXCOORD6;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
};

void main(in VSInput In, out VSOutput Out, uint instanceId : SV_InstanceID)
{
    uint objectIndex = g_InstanceIndex;
    uint cameraIndex = g_CurrentCameraIndex;
    uint colorLayer = 0u;
    uint shadowLayer = CRESSIM_INVALID_BATCH_CAMERA_LAYER;
    uint mainLightIndex = CRESSIM_INVALID_GPU_SCENE_INDEX;
    if (g_UseDrawListBuffer != 0u)
    {
        const VisiblePairInstance pair = g_VisiblePairs[g_DrawListOffset + instanceId];
        const BatchCameraMetadata batchCamera = g_BatchCameras[pair.batchCameraIndex];
        objectIndex = pair.objectIndex;
        cameraIndex = batchCamera.globalCameraIndex;
        colorLayer = batchCamera.colorLayer;
        shadowLayer = batchCamera.shadowLayer;
        mainLightIndex = batchCamera.mainLightIndex;
    }
    const PreparedCamera preparedCamera = g_PreparedCameras[cameraIndex];
    bool poseValid = false;
    float3 position = float3(0.0, 0.0, 0.0);
    float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
    float3 scale = float3(1.0, 1.0, 1.0);
    loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
    const uint localObjectIndex = objectIndex - preparedCamera.objectRangeStart;
    const uint visibilityIndex = preparedCamera.visibilityDataOffset + localObjectIndex;
    if (!poseValid || g_RenderableVisibilityFlags[visibilityIndex] == 0u ||
        preparedCamera.active == 0u)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.WorldPos = float3(0.0, 0.0, 0.0);
        Out.WorldNormal = float3(0.0, 1.0, 0.0);
        Out.TexCoord = In.TexCoord;
        Out.WorldTangent = float4(1.0, 0.0, 0.0, 1.0);
        Out.CameraIndex = cameraIndex;
        Out.MainLightIndex = mainLightIndex;
        Out.ShadowLayer = shadowLayer;
#if MANUAL_LAYER_EXPORT
        Out.Layer = colorLayer;
#endif
        return;
    }

    const float4 worldPos =
        float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
    float3 safeScale = max(abs(scale), float3(1e-6, 1e-6, 1e-6));
    const float3 worldNormal = normalize(quaternionRotateVector(orientation, In.Normal / safeScale));
    float3 worldTangent =
        quaternionRotateVector(orientation, In.Tangent.xyz * scale);
    worldTangent = normalize(worldTangent - worldNormal * dot(worldNormal, worldTangent));
    const float transformSign = (scale.x * scale.y * scale.z) < 0.0 ? -1.0 : 1.0;

    Out.Position = mul(worldPos, preparedCamera.viewProjectionMatrix);
    Out.WorldPos = worldPos.xyz;
    Out.WorldNormal = worldNormal;
    Out.TexCoord = In.TexCoord;
    Out.WorldTangent = float4(worldTangent, In.Tangent.w * transformSign);
    Out.CameraIndex = cameraIndex;
    Out.MainLightIndex = mainLightIndex;
    Out.ShadowLayer = shadowLayer;
#if MANUAL_LAYER_EXPORT
    Out.Layer = colorLayer;
#endif
}
