#ifndef CRESSIM_NEO_PHYSICS_SCAN_CONSTANTS_HLSLI
#define CRESSIM_NEO_PHYSICS_SCAN_CONSTANTS_HLSLI

cbuffer PhysicsScanDispatchConstantsBuffer
{
    uint scanLevelIndex;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

CRESSIM_STRUCTURED_BUFFER(uint4, g_ScanConstants);

uint ScanElementCount()
{
    return CRESSIM_SB_LOAD(g_ScanConstants, scanLevelIndex).x;
}

bool ScanHasParentOffsets()
{
    return CRESSIM_SB_LOAD(g_ScanConstants, scanLevelIndex).y != 0u;
}

#endif // CRESSIM_NEO_PHYSICS_SCAN_CONSTANTS_HLSLI
