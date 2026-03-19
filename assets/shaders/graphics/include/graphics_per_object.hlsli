#ifndef CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI
#define CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI

cbuffer GraphicsPerObject
{
    uint g_InstanceIndex;
    uint g_DrawListOffset;
    uint g_UseDrawListBuffer;
    uint g_PerObjectPadding0;
};

#endif // !CRESSIM_NEO_GRAPHICS_PER_OBJECT_HLSLI
