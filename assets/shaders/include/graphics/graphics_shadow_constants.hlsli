#ifndef CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI
#define CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI

#include "graphics/graphics_per_object.hlsli"

#if !defined(CRESSIM_CAMERA_DEPTH_PASS)
cbuffer GraphicsShadowPerPass
{
    uint4 g_ShadowPassParams;
};

#define g_ShadowMatrixIndex g_ShadowPassParams.x
#define g_ShadowPassMode g_ShadowPassParams.y
#define g_CascadeIndex g_ShadowMatrixIndex
#endif

#endif // CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI
