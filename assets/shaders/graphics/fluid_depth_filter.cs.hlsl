cbuffer GraphicsFluidDepthFilter
{
    uint4 g_FilterParams;
    float4 g_FilterMisc;
};

#define g_FilterLayer g_FilterParams.x
#define g_FilterWidth g_FilterParams.y
#define g_FilterHeight g_FilterParams.z
#define g_FilterRadius g_FilterParams.w
#define g_FilterDepthThreshold g_FilterMisc.x

Texture2DArray<float> g_SourceDepth;
RWTexture2DArray<float> g_FilteredDepthRW;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= g_FilterWidth || pixel.y >= g_FilterHeight)
    {
        return;
    }

    const int3 coord = int3(pixel, g_FilterLayer);
    const float centerDepth = g_SourceDepth.Load(int4(coord, 0));
    if (centerDepth > 999999.0)
    {
        g_FilteredDepthRW[coord] = centerDepth;
        return;
    }

    float weightedDepth = 0.0;
    float totalWeight = 0.0;
    [loop]
    for (int y = -int(g_FilterRadius); y <= int(g_FilterRadius); ++y)
    {
        [loop]
        for (int x = -int(g_FilterRadius); x <= int(g_FilterRadius); ++x)
        {
            const int2 samplePixel = int2(pixel) + int2(x, y);
            if (samplePixel.x < 0 || samplePixel.y < 0 ||
                samplePixel.x >= int(g_FilterWidth) || samplePixel.y >= int(g_FilterHeight))
            {
                continue;
            }

            const float sampleDepth =
                g_SourceDepth.Load(int4(int3(samplePixel, g_FilterLayer), 0));
            if (sampleDepth > 999999.0)
            {
                continue;
            }

            const float spatial = float(x * x + y * y);
            const float range = abs(sampleDepth - centerDepth);
            const float w = exp(-spatial * 0.2 - range / max(g_FilterDepthThreshold, 1.0e-4));
            weightedDepth += sampleDepth * w;
            totalWeight += w;
        }
    }

    g_FilteredDepthRW[coord] = totalWeight > 0.0 ? weightedDepth / totalWeight : centerDepth;
}
