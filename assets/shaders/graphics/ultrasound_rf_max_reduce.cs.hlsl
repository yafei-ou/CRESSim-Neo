#include "structured_buffer_compat.hlsli"

cbuffer UltrasoundReductionConstants
{
    uint g_DataLength;
    uint g_Padding0;
    uint g_Padding1;
    uint g_Padding2;
};

CRESSIM_STRUCTURED_BUFFER(float2, g_RfData);
RWStructuredBuffer<float> g_GroupMaxRW;

groupshared float g_SharedMax[256];

float computeMagnitude(uint index)
{
    const float2 sample = CRESSIM_SB_LOAD(g_RfData, index);
    return length(sample);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupID : SV_GroupID,
          uint groupIndex : SV_GroupIndex)
{
    const uint index = dispatchThreadID.x;
    float value = 0.0f;
    if (index < g_DataLength)
    {
        value = computeMagnitude(index);
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
        g_GroupMaxRW[groupID.x] = g_SharedMax[0];
    }
}
