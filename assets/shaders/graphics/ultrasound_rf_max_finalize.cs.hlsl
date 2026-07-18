#include "structured_buffer_compat.hlsli"

cbuffer UltrasoundReductionConstants
{
    uint g_DataLength;
    uint g_Padding0;
    uint g_Padding1;
    uint g_Padding2;
};

StructuredBuffer<float> g_GroupMax;
RWStructuredBuffer<float> g_FinalMaxRW;

groupshared float g_SharedMax[256];

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    float value = 0.0f;
    for (uint index = dispatchThreadID.x; index < g_DataLength; index += 256u)
    {
        value = max(value, g_GroupMax[index]);
    }

    g_SharedMax[groupIndex] = value;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (groupIndex < stride)
        {
            g_SharedMax[groupIndex] = max(g_SharedMax[groupIndex], g_SharedMax[groupIndex + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (groupIndex == 0u)
    {
        g_FinalMaxRW[0] = g_SharedMax[0];
    }
}
