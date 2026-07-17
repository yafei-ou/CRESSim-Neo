cbuffer UltrasoundScanlineUpdateConstantsBuffer
{
    uint g_EntityPoseSlot;
    uint g_ScanlineCount;
    uint g_Padding0;
    uint g_Padding1;
};

struct PackedScanline
{
    float4 origin;
    float4 radialDirection;
    float4 lateralDirection;
    float4 elevationalDirection;
};

#include "structured_buffer_compat.hlsli"

StructuredBuffer<float4> g_EntityPositions;
StructuredBuffer<float4> g_EntityOrientations;
StructuredBuffer<PackedScanline> g_LocalScanlines;
RWStructuredBuffer<PackedScanline> g_WorldScanlinesRW;

float3 quat_rotate(float4 q, float3 v)
{
    float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= g_ScanlineCount)
    {
        return;
    }

    PackedScanline localScanline = g_LocalScanlines[index];
    PackedScanline worldScanline = localScanline;
    const float3 probePosition = g_EntityPositions[g_EntityPoseSlot].xyz;
    const float4 probeRotation = normalize(g_EntityOrientations[g_EntityPoseSlot]);
    worldScanline.origin.xyz =
        quat_rotate(probeRotation, localScanline.origin.xyz) + probePosition;
    worldScanline.radialDirection.xyz =
        normalize(quat_rotate(probeRotation, localScanline.radialDirection.xyz));
    worldScanline.lateralDirection.xyz =
        normalize(quat_rotate(probeRotation, localScanline.lateralDirection.xyz));
    worldScanline.elevationalDirection.xyz =
        normalize(quat_rotate(probeRotation, localScanline.elevationalDirection.xyz));
    worldScanline.origin.w = 0.0f;
    worldScanline.radialDirection.w = 0.0f;
    worldScanline.lateralDirection.w = 0.0f;
    worldScanline.elevationalDirection.w = 0.0f;
    g_WorldScanlinesRW[index] = worldScanline;
}
