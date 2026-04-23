#ifndef CRESSIM_NEO_PHYSICS_SOFT_RENDER_TYPES_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_RENDER_TYPES_HLSLI

#include "../core/physics_base.hlsli"

struct GpuSoftRenderVertexTriangleRange
{
    uint start;
    uint count;
    uint reserved0;
    uint reserved1;
};

#endif // CRESSIM_NEO_PHYSICS_SOFT_RENDER_TYPES_HLSLI
