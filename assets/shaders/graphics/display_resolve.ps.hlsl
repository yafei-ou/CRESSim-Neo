cbuffer GraphicsDisplayResolve
{
    uint g_SourceLayer;
    uint g_OutputMode;
    uint g_ToneMapper;
    uint g_SourceIsDisplayEncoded;
    float g_Exposure;
    float3 g_Padding;
};

Texture2DArray<float4> g_SourceColor;
SamplerState g_SourceColor_sampler;

struct PSInput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
};

float interleavedGradientNoise(float2 pixelCoord)
{
    const float magicX = 0.06711056;
    const float magicY = 0.00583715;
    const float magicZ = 52.9829189;
    return frac(magicZ * frac(dot(pixelCoord, float2(magicX, magicY))));
}

float3 toneMapReinhard(float3 color)
{
    return color / (1.0 + color);
}

float3 toneMapFilmic(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
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
    color.rgb = max(color.rgb, 0.0) * max(g_Exposure, 0.0);
    if (g_OutputMode == 0 && g_SourceIsDisplayEncoded == 0)
    {
        if (g_ToneMapper == 1)
        {
            color.rgb = toneMapReinhard(color.rgb);
        }
        else if (g_ToneMapper == 2)
        {
            color.rgb = toneMapFilmic(color.rgb);
        }
        color.rgb = linearToSrgb(color.rgb);
        const float dither = interleavedGradientNoise(In.Position.xy) - 0.5;
        color.rgb = saturate(color.rgb + dither / 255.0);
    }
    else if (g_OutputMode == 0)
    {
        color.rgb = saturate(color.rgb);
    }
    return color;
}
