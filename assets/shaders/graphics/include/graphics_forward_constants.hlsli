#ifndef CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI
#define CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI

#include "graphics/include/graphics_per_object.hlsli"

cbuffer GriphicsForwardPerFrame
{
    float g_HasAnyShadowMap;
    float g_ShadowMinimumVisibility;
    float2 g_ForwardShadowPadding;
    float4 g_IblSpecularParams;
    uint4 g_FrameParams;
};

#define g_CurrentCameraIndex g_FrameParams.x

cbuffer GraphicsForwardPerMaterial
{
    float4 g_BaseColorFactor;
    float4 g_EmissiveFactor;
    float4 g_MaterialParams;
};

#endif // CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI
