#include "graphics/include/shadow_constants.hlsli"

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
    float4 worldPos = mul(float4(In.Position, 1.0), g_Model);
    Out.Position = mul(worldPos, g_LightViewProjection);
}
