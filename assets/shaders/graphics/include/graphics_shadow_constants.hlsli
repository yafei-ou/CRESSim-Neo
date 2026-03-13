#ifndef CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI
#define CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI

#include "graphics/include/graphics_per_object.hlsli"

cbuffer GraphicsShadowPerPass
{
    float4x4 g_LightViewProjection;
};

#endif // !CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI