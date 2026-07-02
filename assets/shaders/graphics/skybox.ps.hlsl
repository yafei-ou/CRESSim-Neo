#include "include/structured_buffer_compat.hlsli"
#include "include/graphics/graphics_camera_input.hlsli"
#include "include/graphics/graphics_scene_buffers.hlsli"

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
    nointerpolation uint CameraIndex : TEXCOORD1;
    nointerpolation uint EnvIndex : TEXCOORD2;
};

float degreesToRadians(float degrees)
{
    return degrees * 0.017453292519943295f;
}

float computeEffectiveViewportAspect(float4 viewportAndOutputSize)
{
    const float viewportWidth = clamp(viewportAndOutputSize.x, 0.0, 1.0);
    const float viewportHeight = clamp(viewportAndOutputSize.y, 0.0, 1.0);
    const float outputWidth = max(viewportAndOutputSize.z, 1.0);
    const float outputHeight = max(viewportAndOutputSize.w, 1.0);
    const float effectiveWidth = outputWidth * max(viewportWidth, 1.0e-5);
    const float effectiveHeight = outputHeight * max(viewportHeight, 1.0e-5);
    return max(effectiveWidth / max(effectiveHeight, 1.0e-5), 1.0e-5);
}

float4 main(in PSInput In) : SV_Target
{
    const CameraInput camera = CRESSIM_SB_LOAD(g_CameraInputs, In.CameraIndex);
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, In.CameraIndex);
    const EnvironmentBackgroundLookupEntry entry =
        CRESSIM_SB_LOAD(g_EnvironmentBackgroundLookup, In.EnvIndex);
    if (camera.active == 0u || preparedCamera.active == 0u || entry.enabled == 0u)
    {
        discard;
    }

    const float2 ndc = float2(In.TexCoord.x * 2.0 - 1.0, 1.0 - In.TexCoord.y * 2.0);
    const float fovRadians =
        max(degreesToRadians(camera.projectionParams.x), degreesToRadians(1.0));
    const float tanHalfFov = tan(0.5 * fovRadians);
    const float viewportAspect = computeEffectiveViewportAspect(camera.viewportAndOutputSize);
    const float3 viewDir =
        normalize(float3(ndc.x * max(viewportAspect, 1.0e-5) * tanHalfFov,
                         ndc.y * tanHalfFov, 1.0));
    const float3x3 worldFromView = transpose((float3x3)preparedCamera.viewMatrix);
    const float3 dir = normalize(mul(viewDir, worldFromView));

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
