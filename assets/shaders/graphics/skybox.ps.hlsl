#include "include/structured_buffer_compat.hlsli"

struct CameraInput
{
    float4 position;
    float4 orientation;
    float4 projectionParams;
    float4 viewportAndOutputSize;
    uint envIndex;
    uint cameraSlot;
    uint active;
    uint reserved;
};

cbuffer GraphicsSkybox
{
    uint g_SkyboxCameraIndex;
    uint g_SkyboxTargetLayer;
    float g_SkyboxViewportAspect;
    float g_SkyboxPadding0;
};

struct EnvironmentBackgroundLookupEntry
{
    uint sliceIndex;
    uint enabled;
    float intensity;
    float reserved0;
};

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_STRUCTURED_BUFFER(EnvironmentBackgroundLookupEntry, g_EnvironmentBackgroundLookup);
Texture2DArray g_EnvironmentBackgroundArray;
SamplerState g_EnvironmentBackgroundArray_sampler;

struct PSInput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
    nointerpolation uint EnvIndex : TEXCOORD1;
};

float degreesToRadians(float degrees)
{
    return degrees * 0.017453292519943295f;
}

float3 quaternionRotateVector(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float4 main(in PSInput In) : SV_Target
{
    const CameraInput camera = CRESSIM_SB_LOAD(g_CameraInputs, g_SkyboxCameraIndex);
    const EnvironmentBackgroundLookupEntry entry =
        CRESSIM_SB_LOAD(g_EnvironmentBackgroundLookup, In.EnvIndex);
    if (camera.active == 0u || entry.enabled == 0u)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    const float2 ndc = float2(In.TexCoord.x * 2.0 - 1.0, 1.0 - In.TexCoord.y * 2.0);
    const float fovRadians =
        max(degreesToRadians(camera.projectionParams.x), degreesToRadians(1.0));
    const float tanHalfFov = tan(0.5 * fovRadians);
    const float3 viewDir =
        normalize(float3(ndc.x * max(g_SkyboxViewportAspect, 1.0e-5) * tanHalfFov,
                         ndc.y * tanHalfFov, 1.0));
    const float3 dir = normalize(quaternionRotateVector(normalize(camera.orientation), viewDir));

    // Match the CPU baker's cubemap face convention exactly so the visible background aligns
    // with the environment used for irradiance/specular IBL generation.
    const float3 absDir = abs(dir);
    uint face = 0u;
    float u = 0.5;
    float v = 0.5;

    if (absDir.x >= absDir.y && absDir.x >= absDir.z)
    {
        if (dir.x >= 0.0)
        {
            face = 0u;
            u = -dir.z / absDir.x;
            v = -dir.y / absDir.x;
        }
        else
        {
            face = 1u;
            u = dir.z / absDir.x;
            v = -dir.y / absDir.x;
        }
    }
    else if (absDir.y >= absDir.x && absDir.y >= absDir.z)
    {
        if (dir.y >= 0.0)
        {
            face = 2u;
            u = dir.x / absDir.y;
            v = dir.z / absDir.y;
        }
        else
        {
            face = 3u;
            u = dir.x / absDir.y;
            v = -dir.z / absDir.y;
        }
    }
    else
    {
        if (dir.z >= 0.0)
        {
            face = 4u;
            u = dir.x / absDir.z;
            v = -dir.y / absDir.z;
        }
        else
        {
            face = 5u;
            u = -dir.x / absDir.z;
            v = -dir.y / absDir.z;
        }
    }

    u = saturate(u * 0.5 + 0.5);
    v = saturate(v * 0.5 + 0.5);
    const float layer = (float)(entry.sliceIndex * 6u + face);
    const float3 color =
        g_EnvironmentBackgroundArray.Sample(g_EnvironmentBackgroundArray_sampler,
                                            float3(u, v, layer)).rgb *
        max(entry.intensity, 0.0);
    return float4(color, 1.0);
}
