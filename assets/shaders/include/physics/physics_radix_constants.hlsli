#ifndef CRESSIM_NEO_PHYSICS_RADIX_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_RADIX_CONSTANTS_HLSLI

#include "../structured_buffer_compat.hlsli"

cbuffer PhysicsRadixConstantsBuffer
{
    uint elementCount;
    uint bitIndex;
    uint reserved0;
    uint reserved1;
};

#endif // CRESSIM_NEO_PHYSICS_RADIX_CONSTANTS_HLSLI
