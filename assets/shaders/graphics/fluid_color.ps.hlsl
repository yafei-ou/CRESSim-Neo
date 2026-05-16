#include "include/structured_buffer_compat.hlsli"

struct CameraInput
{
    float4 position;
    float4 orientation;
    float4 projectionParams;
    float4 viewportAndOutputSize;
    uint envIndex;
    uint cameraSlot;
    uint active;
    uint reserved;
};

cbuffer GraphicsFluidColor
{
    uint4 g_FluidColorParams;
};

#define g_FluidColorCameraIndex g_FluidColorParams.x
#define g_FluidColorDepthLayer g_FluidColorParams.y

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
Texture2DArray<float> g_FilteredFluidDepth;
SamplerState g_FilteredFluidDepth_sampler;

struct PSInput
{
    float4 Position : SV_Position;
    float3 ViewPos : TEXCOORD0;
    float2 Uv : TEXCOORD1;
    float4 Color : TEXCOORD2;
};

float mainDepth(float depth, float nearClip, float farClip)
{
    const float zScale = farClip / max(farClip - nearClip, 1.0e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1.0e-5);
    return zScale + zTranslate / max(depth, 1.0e-5);
}

float4 main(PSInput In) : SV_Target
{
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_FluidColorCameraIndex);
    const float2 uv = In.Position.xy / cameraInput.viewportAndOutputSize.zw;
    const float fluidDepth = g_FilteredFluidDepth.SampleLevel(
        g_FilteredFluidDepth_sampler, float3(uv, g_FluidColorDepthLayer), 0.0);
    const float depth = In.ViewPos.z;
    if (fluidDepth > 999999.0 || abs(fluidDepth - depth) > max(0.02, depth * 0.02))
    {
        discard;
    }

    const float2 distVec = In.Uv - float2(0.5, 0.5);
    if (dot(distVec, distVec) > 0.25)
    {
        discard;
    }

    return In.Color;
}
