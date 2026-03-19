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
    const uint shadowMask =
        g_RenderableShadowCascadeMasks[preparedCamera.renderableDataOffset + objectIndex];
    if (!poseValid || preparedCamera.active == 0u || ((shadowMask & (1u << g_CascadeIndex)) == 0u))
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    const float4 worldPos =
        float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
    Out.Position = mul(worldPos, preparedCamera.lightViewProjectionMatrices[g_CascadeIndex]);
}
