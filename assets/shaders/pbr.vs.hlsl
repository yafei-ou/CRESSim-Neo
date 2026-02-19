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
    float4 ShadowPos : TEXCOORD2;
};

cbuffer PbrConstants
{
    float4x4 g_Model;
    float4x4 g_ViewProjection;
    float4x4 g_LightViewProjection;
    float4 g_CameraPositionMetallic;
    float4 g_LightDirectionIntensity;
    float4 g_LightColorRoughness;
    float4 g_BaseColor;
    float4 g_ShadowParams;
};

void main(in VSInput In, out VSOutput Out)
{
    float4 worldPos = mul(float4(In.Position, 1.0), g_Model);
    Out.Position = mul(worldPos, g_ViewProjection);
    Out.WorldPos = worldPos.xyz;
    Out.WorldNormal = normalize(mul(float4(In.Normal, 0.0), g_Model).xyz);
    Out.ShadowPos = mul(worldPos, g_LightViewProjection);
}
