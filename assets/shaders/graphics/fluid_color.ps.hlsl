#include "include/structured_buffer_compat.hlsli"
#include "include/graphics/graphics_camera_input.hlsli"

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
    float DepthTolerance : TEXCOORD3;
};

float4 main(PSInput In) : SV_Target
{
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_FluidColorCameraIndex);
    const float2 uv = In.Position.xy / cameraInput.viewportAndOutputSize.zw;
    const float fluidDepth = g_FilteredFluidDepth.SampleLevel(
        g_FilteredFluidDepth_sampler, float3(uv, g_FluidColorDepthLayer), 0.0);
    const float depth = In.ViewPos.z;
    if (fluidDepth > 999999.0 || abs(fluidDepth - depth) > In.DepthTolerance)
    {
        discard;
    }

    const float2 distVec = In.Uv - float2(0.5, 0.5);
    if (dot(distVec, distVec) > 0.36)
    {
        discard;
    }

    return In.Color;
}
