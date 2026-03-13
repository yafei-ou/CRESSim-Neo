#ifndef CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI
#define CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI

#include "graphics/include/graphics_per_object.hlsli"

cbuffer GriphicsForwardPerFrame
{
    float4x4 g_ViewMatrix;
    float4x4 g_ViewProjection;
    float4x4 g_LightViewProjection[4];
    float4 g_CameraPosition;
    float4 g_LightDirectionIntensity;
    float4 g_LightColor;
    float4 g_CascadeSplits;
    float4 g_ShadowTexelSizeCascadeCount;
    float4 g_ShadowParams;
};

cbuffer CressimForwardPerMaterial
{
    float4 g_BaseColor;
    float4 g_MaterialParams;
};

#endif // !CRESSIM_NEO_GRAPHICS_FORWARD_CONSTANTS_HLSLI