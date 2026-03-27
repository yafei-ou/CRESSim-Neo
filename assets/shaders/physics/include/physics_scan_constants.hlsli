#ifndef CRESSIM_NEO_PHYSICS_SCAN_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_SCAN_CONSTANTS_HLSLI

#include "include/structured_buffer_compat.hlsli"

cbuffer PhysicsScanConstantsBuffer
{
    uint elementCount;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

#endif // CRESSIM_NEO_PHYSICS_SCAN_CONSTANTS_HLSLI
