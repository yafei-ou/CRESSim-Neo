#ifndef CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI
#define CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI

cbuffer GraphicsPerObject
{
    float4x4 g_Model;
    float4x4 g_NormalMatrix;
    uint g_InstanceIndex;
    uint g_UseSceneBuffers;
    uint g_ObjectPadding0;
    uint g_ObjectPadding1;
};

#endif // !CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI
