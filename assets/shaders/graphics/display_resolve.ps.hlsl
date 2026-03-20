cbuffer GraphicsDisplayResolve
{
    uint g_SourceLayer;
    uint g_Padding0;
    uint g_Padding1;
    uint g_Padding2;
};

Texture2DArray<float4> g_SourceColor;
SamplerState g_SourceColor_sampler;

struct PSInput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
};

float4 main(in PSInput In) : SV_Target
{
    return g_SourceColor.Sample(g_SourceColor_sampler, float3(In.TexCoord, (float)g_SourceLayer));
}
