#include "physics/include/physics_scan_constants.hlsli"

StructuredBuffer<uint> g_ScannedBlockOffsets;
RWStructuredBuffer<uint> g_ScanOutput;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID,
                                 uint3 groupID : SV_GroupID)
{
    const uint index = dispatchThreadID.x;
    if (index >= elementCount)
    {
        return;
    }

    g_ScanOutput[index] += g_ScannedBlockOffsets[groupID.x];
}
