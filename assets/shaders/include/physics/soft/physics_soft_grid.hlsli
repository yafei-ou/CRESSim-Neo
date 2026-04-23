#ifndef CRESSIM_NEO_PHYSICS_SOFT_GRID_HLSLI
#define CRESSIM_NEO_PHYSICS_SOFT_GRID_HLSLI

#include "physics_soft_types.hlsli"

uint ComputeParticleGridCellKey(int gx, int gy, int gz)
{
    uint seed = uint(gx) * 73856093u;
    seed ^= uint(gy) * 19349663u;
    seed ^= uint(gz) * 83492791u;
    return seed;
}

#endif // CRESSIM_NEO_PHYSICS_SOFT_GRID_HLSLI
