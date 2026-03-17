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

void main(in VSInput In, out VSOutput Out)
{
    float4 worldPos = float4(0.0, 0.0, 0.0, 1.0);
    if (g_UseSceneBuffers != 0u)
    {
        bool poseValid = false;
        float3 position = float3(0.0, 0.0, 0.0);
        float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
        float3 scale = float3(1.0, 1.0, 1.0);
        loadRenderablePose(g_InstanceIndex, poseValid, position, orientation, scale);
        const uint shadowMask = g_RenderableShadowCascadeMasks[g_InstanceIndex];
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
    }
    Out.Position = mul(worldPos, g_LightViewProjection);
}
