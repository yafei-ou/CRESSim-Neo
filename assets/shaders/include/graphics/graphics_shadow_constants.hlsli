#ifndef CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI
#define CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI

#include "graphics_per_object.hlsli"

cbuffer GraphicsShadowPerPass
{
    uint4 g_ShadowPassParams;
};

#define g_ShadowMatrixIndex g_ShadowPassParams.x
#define g_ShadowPassMode g_ShadowPassParams.y
#define g_CascadeIndex g_ShadowMatrixIndex

#endif // CRESSIM_NEO_GRAPHICS_SHADOW_CONSTANTS_HLSLI
