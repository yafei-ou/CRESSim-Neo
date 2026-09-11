#ifndef CRESSIM_NEO_PHYSICS_SURFACE_RENDER_DISPATCH_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_SURFACE_RENDER_DISPATCH_CONSTANTS_HLSLI

cbuffer PhysicsSurfaceRenderDispatchConstantsBuffer
{
    uint renderVertexCount;
    uint renderTriangleCount;
    uint surfaceCount;
    uint surfaceRenderReserved0;
};

#endif // CRESSIM_NEO_PHYSICS_SURFACE_RENDER_DISPATCH_CONSTANTS_HLSLI
