#ifndef CRESSIM_NEO_PHYSICS_SOFT_RENDER_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_RENDER_DISPATCH_CONSTANTS_HLSLI

#include "include/structured_buffer_compat.hlsli"

cbuffer PhysicsSoftRenderDispatchConstantsBuffer
{
    uint renderVertexCount;
    uint renderTriangleCount;
    uint softBodyCount;
    uint softRenderReserved0;
};

#endif // CRESSIM_NEO_PHYSICS_SOFT_RENDER_DISPATCH_CONSTANTS_HLSLI
