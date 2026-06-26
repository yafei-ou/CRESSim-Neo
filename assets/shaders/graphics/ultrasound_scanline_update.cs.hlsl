cbuffer UltrasoundScanlineUpdateConstantsBuffer
{
    float4 g_ProbePosition;
    float4 g_ProbeRotation;
    uint g_ScanlineCount;
    uint g_Padding0;
    uint g_Padding1;
    uint g_Padding2;
};

struct PackedScanline
{
    float4 origin;
    float4 radialDirection;
    float4 lateralDirection;
    float4 elevationalDirection;
};

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
    worldScanline.origin.xyz =
        quat_rotate(g_ProbeRotation, localScanline.origin.xyz) + g_ProbePosition.xyz;
    worldScanline.radialDirection.xyz =
        normalize(quat_rotate(g_ProbeRotation, localScanline.radialDirection.xyz));
    worldScanline.lateralDirection.xyz =
        normalize(quat_rotate(g_ProbeRotation, localScanline.lateralDirection.xyz));
    worldScanline.elevationalDirection.xyz =
        normalize(quat_rotate(g_ProbeRotation, localScanline.elevationalDirection.xyz));
    worldScanline.origin.w = 0.0f;
    worldScanline.radialDirection.w = 0.0f;
    worldScanline.lateralDirection.w = 0.0f;
    worldScanline.elevationalDirection.w = 0.0f;
    g_WorldScanlinesRW[index] = worldScanline;
}
