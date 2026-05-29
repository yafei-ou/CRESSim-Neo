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
    float g_SectorAngleRadians;
    float g_ProbeRadiusPixels;
    float g_Padding0;
    float g_Padding1;
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

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= g_ImageWidth || dispatchThreadID.y >= g_ImageHeight)
    {
        return;
    }

    if (g_NumScanlines == 0u || g_SamplesPerLine == 0u || g_ImageWidth == 0u || g_ImageHeight == 0u)
    {
        g_OutputImageRW[dispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float halfAngle = 0.5f * max(g_SectorAngleRadians, 1.0e-6f);
    const float angleStep = g_NumScanlines > 1u ? g_SectorAngleRadians / float(g_NumScanlines - 1u)
                                                : 0.0f;
    const float shiftedX = (float(dispatchThreadID.x) + 0.5f) - 0.5f * float(g_ImageWidth);
    const float verticalOffset = g_ProbeRadiusPixels * cos(halfAngle);
    const float polarY = (float(dispatchThreadID.y) + 0.5f) + verticalOffset;
    const float actualAngle = atan2(shiftedX, polarY);
    const float actualScanlineIdx =
        angleStep > 0.0f ? (actualAngle + halfAngle) / angleStep : 0.0f;
    const int scanlineIdx = int(floor(actualScanlineIdx));
    const float radius = sqrt(shiftedX * shiftedX + polarY * polarY);

    if (scanlineIdx < 0 || scanlineIdx >= int(g_NumScanlines) ||
        radius < g_ProbeRadiusPixels ||
        radius > g_ProbeRadiusPixels + float(g_SamplesPerLine))
    {
        g_OutputImageRW[dispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float distanceToOrigin = radius - g_ProbeRadiusPixels;
    const int sampleIdx = int(floor(distanceToOrigin));
    if (sampleIdx < 0 || sampleIdx >= int(g_SamplesPerLine))
    {
        g_OutputImageRW[dispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const uint s0 = uint(scanlineIdx);
    const uint s1 = min(s0 + 1u, g_NumScanlines - 1u);
    const uint r0 = uint(sampleIdx);
    const uint r1 = min(r0 + 1u, g_SamplesPerLine - 1u);
    const float wScanline = actualScanlineIdx - float(scanlineIdx);
    const float wRadial = distanceToOrigin - float(sampleIdx);

    const float mag00 = sampleMagnitude(s0, r0);
    const float mag10 = sampleMagnitude(s1, r0);
    const float mag01 = sampleMagnitude(s0, r1);
    const float mag11 = sampleMagnitude(s1, r1);

    const float mag0 = lerp(mag00, mag10, wScanline);
    const float mag1 = lerp(mag01, mag11, wScanline);
    const float magnitude = lerp(mag0, mag1, wRadial);

    const float maxMagnitude = g_UseFixedMaxNormalization != 0u
        ? max(g_FixedMaxSignal, 1.0e-6f)
        : max(g_FinalMax[0], 1.0e-6f);
    const float gray = saturate(magnitude / maxMagnitude);
    g_OutputImageRW[dispatchThreadID.xy] = float4(gray, gray, gray, 1.0f);
}
