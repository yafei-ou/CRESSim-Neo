#include "graphics/include/per_object.hlsli"

cbuffer CressimForwardPerFrame
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
