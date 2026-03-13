#ifndef CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI
#define CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI

cbuffer GraphicsPerObject
{
    float4x4 g_Model;
    float4x4 g_NormalMatrix;
};

#endif // !CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI