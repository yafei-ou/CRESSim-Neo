cbuffer GraphicsBatchPresent
{
    uint4 g_BatchPresentParams;
};

#define g_BatchLayer g_BatchPresentParams.x

Texture2DArray g_BatchColor;
SamplerState g_BatchColor_sampler;

struct VSOutput
{
    float4 Position : SV_Position;
    float2 Uv : TEXCOORD0;
};

float4 main(VSOutput input) : SV_Target
{
    return g_BatchColor.Sample(g_BatchColor_sampler, float3(input.Uv, (float)g_BatchLayer));
}
