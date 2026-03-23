#ifndef CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI
#define CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI

#include "graphics/include/graphics_per_object.hlsli"

cbuffer GriphicsForwardPerFrame
{
    float4 g_ShadowParams;
    uint4 g_FrameParams;
};

#define g_CurrentCameraIndex g_FrameParams.x

cbuffer GraphicsForwardPerMaterial
{
    float4 g_BaseColor;
    float4 g_MaterialParams;
};

#endif // CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI
