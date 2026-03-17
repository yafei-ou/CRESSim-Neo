#ifndef CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI
#define CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI

#include "graphics/include/graphics_per_object.hlsli"

cbuffer GraphicsShadowPerPass
{
    float4x4 g_LightViewProjection;
    uint4 g_ShadowPassParams;
};

#define g_CascadeIndex g_ShadowPassParams.x
#define g_CurrentCameraIndex g_ShadowPassParams.y

#endif // !CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI
