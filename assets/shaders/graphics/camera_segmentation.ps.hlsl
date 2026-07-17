#include "graphics/graphics_forward_constants.hlsli"

Texture2D g_BaseColorTexture;
SamplerState g_BaseColorTexture_sampler;

struct PSInput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD7;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
    nointerpolation uint SegmentationId : TEXCOORD0;
};

uint main(in PSInput In) : SV_Target
{
#ifdef CRESSIM_FEATURE_ALPHA_TEST
    const float4 sampledBaseColor = g_BaseColorTexture.Sample(g_BaseColorTexture_sampler, In.TexCoord);
    const float alpha = g_BaseColorFactor.a * sampledBaseColor.a;
    if (alpha < g_MaterialParams.z)
    {
        discard;
    }
#endif
    return In.SegmentationId;
}
