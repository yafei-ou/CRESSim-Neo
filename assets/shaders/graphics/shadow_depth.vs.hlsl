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
    float4 worldPos = float4(0.0, 0.0, 0.0, 1.0);
    float4x4 lightViewProjection = float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
    if (g_UseSceneBuffers != 0u)
    {
        uint objectIndex = g_InstanceIndex;
        if (g_UseDrawListBuffer != 0u)
        {
            objectIndex = g_VisibleObjectIndices[g_DrawListOffset + instanceId];
        }
        const PreparedCamera preparedCamera = g_PreparedCameras[g_CurrentCameraIndex];
        lightViewProjection = preparedCamera.lightViewProjectionMatrices[g_CascadeIndex];
        bool poseValid = false;
        float3 position = float3(0.0, 0.0, 0.0);
        float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
        float3 scale = float3(1.0, 1.0, 1.0);
        loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
        const uint shadowMask = g_RenderableShadowCascadeMasks[objectIndex];
        if (!poseValid || ((shadowMask & (1u << g_CascadeIndex)) == 0u))
        {
            Out.Position = float4(2.0, 2.0, 2.0, 1.0);
            return;
        }
        worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
    }
    else
    {
        worldPos = mul(float4(In.Position, 1.0), g_Model);
        lightViewProjection = g_LightViewProjection;
    }
    Out.Position = mul(worldPos, lightViewProjection);
}
