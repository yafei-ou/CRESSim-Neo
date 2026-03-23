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

float3 toneMapReinhard(float3 color)
{
    return color / (1.0 + color);
}

float3 linearToSrgb(float3 color)
{
    const float3 cutoff = step(float3(0.0031308, 0.0031308, 0.0031308), color);
    const float3 lower = color * 12.92;
    const float3 higher = 1.055 * pow(abs(color), 1.0 / 2.4) - 0.055;
    return lerp(lower, higher, cutoff);
}

float4 main(in PSInput In) : SV_Target
{
    float4 color = g_SourceColor.Sample(g_SourceColor_sampler, float3(In.TexCoord, (float)g_SourceLayer));
    color.rgb = toneMapReinhard(max(color.rgb, 0.0));
    color.rgb = linearToSrgb(color.rgb);
    return color;
}
