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

void main(in VSInput In, out VSOutput Out)
{
    float4 worldPos = float4(0.0, 0.0, 0.0, 1.0);
    float3 worldNormal = float3(0.0, 1.0, 0.0);
    if (g_UseSceneBuffers != 0u)
    {
        bool poseValid = false;
        float3 position = float3(0.0, 0.0, 0.0);
        float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
        float3 scale = float3(1.0, 1.0, 1.0);
        loadRenderablePose(g_InstanceIndex, poseValid, position, orientation, scale);
        if (!poseValid || g_RenderableVisibilityFlags[g_InstanceIndex] == 0u)
        {
            Out.Position = float4(2.0, 2.0, 2.0, 1.0);
            Out.WorldPos = float3(0.0, 0.0, 0.0);
            Out.WorldNormal = float3(0.0, 1.0, 0.0);
            return;
        }

        worldPos = float4(quaternionRotateVector(orientation, In.Position * scale) + position, 1.0);
        float3 safeScale = max(abs(scale), float3(1e-6, 1e-6, 1e-6));
        worldNormal = normalize(quaternionRotateVector(orientation, In.Normal / safeScale));
    }
    else
    {
        worldPos = mul(float4(In.Position, 1.0), g_Model);
        worldNormal = normalize(mul(float4(In.Normal, 0.0), g_NormalMatrix).xyz);
    }

    Out.Position = mul(worldPos, g_ViewProjection);
    Out.WorldPos = worldPos.xyz;
    Out.WorldNormal = worldNormal;
}
