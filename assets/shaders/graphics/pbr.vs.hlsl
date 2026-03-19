#include "graphics/include/graphics_forward_constants.hlsli"
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
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
};

void main(in VSInput In, out VSOutput Out, uint instanceId : SV_InstanceID)
{
    uint objectIndex = g_InstanceIndex;
    if (g_UseDrawListBuffer != 0u)
    {
        objectIndex = g_VisibleObjectIndices[g_DrawListOffset + instanceId];
    }
    const PreparedCamera preparedCamera = g_PreparedCameras[g_CurrentCameraIndex];
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
        return;
    }

    const float4 worldPos =
        float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
    float3 safeScale = max(abs(scale), float3(1e-6, 1e-6, 1e-6));
    const float3 worldNormal = normalize(quaternionRotateVector(orientation, In.Normal / safeScale));

    Out.Position = mul(worldPos, preparedCamera.viewProjectionMatrix);
    Out.WorldPos = worldPos.xyz;
    Out.WorldNormal = worldNormal;
}
