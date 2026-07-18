#include "structured_buffer_compat.hlsli"
#include "graphics/graphics_camera_input.hlsli"

cbuffer GraphicsFluidDepthFilter
{
    uint g_FilterLayer;
    uint g_FilterCameraIndex;
    uint g_FilterMaxRadius;
    uint g_FilterReserved0;
    float g_FilterWorldRadius;
    float g_FilterDepthThreshold;
    float2 g_FilterReserved1;
    uint4 g_FilterViewportRect;
};

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
Texture2DArray<float> g_SourceDepth;
RWTexture2DArray<float> g_FilteredDepthRW;

static const float kInvalidFluidDepth = 999999.0f;

float degreesToRadians(float degrees)
{
    return degrees * 0.01745329252;
}

uint computeFilterRadius(float centerDepth, CameraInput cameraInput)
{
    const float fovRadians =
        max(degreesToRadians(cameraInput.projectionParams.x), degreesToRadians(1.0f));
    const float projectionPixelsPerUnit =
        0.5f * max(cameraInput.viewportAndOutputSize.w, 1.0f) / tan(0.5f * fovRadians);
    const float pixelRadius =
        max(g_FilterWorldRadius, 1.0e-4f) * projectionPixelsPerUnit / max(centerDepth, 1.0e-4f);
    return clamp((uint)ceil(pixelRadius), 1u, max(g_FilterMaxRadius, 1u));
}

float computeSpatialWeight(float2 offset, float sigma)
{
    const float safeSigma = max(sigma, 1.0e-4f);
    return exp(-dot(offset, offset) / (2.0f * safeSigma * safeSigma));
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint textureWidth = 0u;
    uint textureHeight = 0u;
    uint textureLayers = 0u;
    g_SourceDepth.GetDimensions(textureWidth, textureHeight, textureLayers);

    const uint2 viewportSize = max(g_FilterViewportRect.zw, uint2(1u, 1u));
    const uint2 localPixel = dispatchThreadID.xy;
    if (localPixel.x >= viewportSize.x || localPixel.y >= viewportSize.y)
    {
        return;
    }

    const uint2 pixel = g_FilterViewportRect.xy + localPixel;
    const int3 coord = int3(pixel, g_FilterLayer);
    const float centerDepth = g_SourceDepth.Load(int4(coord, 0));
    if (centerDepth > kInvalidFluidDepth)
    {
        g_FilteredDepthRW[coord] = centerDepth;
        return;
    }

    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_FilterCameraIndex);
    const uint filterRadius = computeFilterRadius(centerDepth, cameraInput);
    const float sigma = max(float(filterRadius) / 3.0f, 0.5f);
    const float depthThreshold = max(g_FilterDepthThreshold, 1.0e-4f);

    float weightedDepth = centerDepth;
    float totalWeight = 1.0f;

    [loop]
    for (int y = -int(filterRadius); y <= int(filterRadius); ++y)
    {
        [loop]
        for (int x = -int(filterRadius); x <= int(filterRadius); ++x)
        {
            if (x == 0 && y == 0)
            {
                continue;
            }

            const int2 samplePixel = int2(pixel) + int2(x, y);
            if (samplePixel.x < int(g_FilterViewportRect.x) ||
                samplePixel.y < int(g_FilterViewportRect.y) ||
                samplePixel.x >= int(g_FilterViewportRect.x + viewportSize.x) ||
                samplePixel.y >= int(g_FilterViewportRect.y + viewportSize.y) ||
                samplePixel.x >= int(textureWidth) || samplePixel.y >= int(textureHeight))
            {
                continue;
            }

            float sampleDepth = g_SourceDepth.Load(int4(int3(samplePixel, g_FilterLayer), 0));
            if (sampleDepth > kInvalidFluidDepth)
            {
                continue;
            }

            const float2 offset = float2((float)x, (float)y);
            const float spatialWeight = computeSpatialWeight(offset, sigma);
            const float depthDelta = sampleDepth - centerDepth;
            const float rangeWeight =
                exp(-(depthDelta * depthDelta) / (2.0f * depthThreshold * depthThreshold));
            const float weight = spatialWeight * rangeWeight;
            weightedDepth += sampleDepth * weight;
            totalWeight += weight;
        }
    }

    g_FilteredDepthRW[coord] = weightedDepth / max(totalWeight, 1.0e-4f);
}
