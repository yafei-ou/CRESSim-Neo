#include "include/structured_buffer_compat.hlsli"

cbuffer UltrasoundImageConstants
{
    uint g_NumScanlines;
    uint g_SamplesPerLine;
    uint g_ImageWidth;
    uint g_ImageHeight;
    float g_LineLength;
    float g_ScanlineSpacing;
    float g_FixedMaxSignal;
    uint g_UseFixedMaxNormalization;
};

CRESSIM_STRUCTURED_BUFFER(float2, g_RfData);
StructuredBuffer<float> g_FinalMax;
RWTexture2D<float4> g_OutputImageRW;

float sampleMagnitude(uint scanline, uint sampleIndex)
{
    if (g_NumScanlines == 0u || g_SamplesPerLine == 0u)
    {
        return 0.0f;
    }

    scanline = min(scanline, g_NumScanlines - 1u);
    sampleIndex = min(sampleIndex, g_SamplesPerLine - 1u);
    const uint linearIndex = scanline * g_SamplesPerLine + sampleIndex;
    const float2 sample = CRESSIM_SB_LOAD(g_RfData, linearIndex);
    return length(sample);
}

float bilinearMagnitude(float lateralCoord, float depthCoord)
{
    const float clampedLateral = clamp(lateralCoord, 0.0f, max(float(g_NumScanlines) - 1.0f, 0.0f));
    const float clampedDepth = clamp(depthCoord, 0.0f, max(float(g_SamplesPerLine) - 1.0f, 0.0f));

    const uint x0 = (uint)floor(clampedLateral);
    const uint x1 = min(x0 + 1u, max(g_NumScanlines, 1u) - 1u);
    const uint y0 = (uint)floor(clampedDepth);
    const uint y1 = min(y0 + 1u, max(g_SamplesPerLine, 1u) - 1u);

    const float tx = clampedLateral - float(x0);
    const float ty = clampedDepth - float(y0);

    const float m00 = sampleMagnitude(x0, y0);
    const float m10 = sampleMagnitude(x1, y0);
    const float m01 = sampleMagnitude(x0, y1);
    const float m11 = sampleMagnitude(x1, y1);

    const float mx0 = lerp(m00, m10, tx);
    const float mx1 = lerp(m01, m11, tx);
    return lerp(mx0, mx1, ty);
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= g_ImageWidth || dispatchThreadID.y >= g_ImageHeight)
    {
        return;
    }

    if (g_NumScanlines == 0u || g_SamplesPerLine == 0u)
    {
        g_OutputImageRW[dispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float normalizedX = (float(dispatchThreadID.x) + 0.5f) / max(float(g_ImageWidth), 1.0f);
    const float normalizedY = (float(dispatchThreadID.y) + 0.5f) / max(float(g_ImageHeight), 1.0f);
    const float lateralCoord = normalizedX * max(float(g_NumScanlines) - 1.0f, 0.0f);
    const float depthCoord = normalizedY * max(float(g_SamplesPerLine) - 1.0f, 0.0f);

    const float magnitude = bilinearMagnitude(lateralCoord, depthCoord);
    const float maxMagnitude = g_UseFixedMaxNormalization != 0u
        ? max(g_FixedMaxSignal, 1.0e-6f)
        : max(g_FinalMax[0], 1.0e-6f);

    const float normalized = saturate(magnitude / maxMagnitude);
    const float gray = normalized;

    const uint flippedY = g_ImageHeight - 1u - dispatchThreadID.y;
    g_OutputImageRW[uint2(dispatchThreadID.x, flippedY)] = float4(gray, gray, gray, 1.0f);
}
