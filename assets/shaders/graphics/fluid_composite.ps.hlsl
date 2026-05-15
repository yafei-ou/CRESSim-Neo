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

cbuffer GraphicsFluidComposite
{
    float4 g_FluidTint;
    float4 g_FluidSpecularSmoothness;
    uint4 g_FluidCompositeParams;
    float4 g_FluidCompositeMisc;
};

#define g_FluidCameraIndex g_FluidCompositeParams.x
#define g_FluidDepthLayer g_FluidCompositeParams.y
#define g_SceneDepthLayer g_FluidCompositeParams.z
#define g_FluidFresnel g_FluidCompositeMisc.x

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
Texture2DArray<float> g_FilteredFluidDepth;
SamplerState g_FilteredFluidDepth_sampler;
Texture2DArray<float> g_SceneDepth;
SamplerState g_SceneDepth_sampler;

struct PSInput
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD0;
};

struct PSOutput
{
    float4 Color : SV_Target;
    float Depth : SV_Depth;
};

float degreesToRadians(float degrees)
{
    return degrees * 0.017453292519943295f;
}

float3 reconstructViewPos(float2 uv, float linearDepth, CameraInput cameraInput)
{
    const float aspect = max((cameraInput.viewportAndOutputSize.x * cameraInput.viewportAndOutputSize.z) /
                             max(cameraInput.viewportAndOutputSize.y * cameraInput.viewportAndOutputSize.w, 1.0e-5),
                             1.0e-5);
    const float fovRadians = max(degreesToRadians(cameraInput.projectionParams.x), degreesToRadians(1.0));
    const float tanHalfFov = tan(0.5 * fovRadians);
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * linearDepth * tanHalfFov * aspect,
                  ndc.y * linearDepth * tanHalfFov,
                  linearDepth);
}

float linearizeDepth(float depth, float nearClip, float farClip)
{
    const float zScale = farClip / max(farClip - nearClip, 1.0e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1.0e-5);
    return zTranslate / max(depth - zScale, -1.0e-5);
}

PSOutput main(PSInput In)
{
    PSOutput Out;
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_FluidCameraIndex);
    const float2 uv = saturate(In.TexCoord);
    const float depth = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                         float3(uv, g_FluidDepthLayer), 0.0);
    if (depth > 999999.0)
    {
        discard;
    }

    const float nearClip = max(cameraInput.projectionParams.y, 1.0e-4);
    const float farClip = max(cameraInput.projectionParams.z, nearClip + 1.0e-4);
    const float sceneDepth = g_SceneDepth.SampleLevel(g_SceneDepth_sampler,
                                                      float3(uv, g_SceneDepthLayer), 0.0);
    const float linearSceneDepth = linearizeDepth(sceneDepth, nearClip, farClip);
    if (depth >= linearSceneDepth)
    {
        discard;
    }

    const float2 pixel = float2(1.0 / max(cameraInput.viewportAndOutputSize.z, 1.0),
                                1.0 / max(cameraInput.viewportAndOutputSize.w, 1.0));
    const float dx = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                      float3(uv + float2(pixel.x, 0.0), g_FluidDepthLayer), 0.0);
    const float dy = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                      float3(uv + float2(0.0, pixel.y), g_FluidDepthLayer), 0.0);
    const float3 centerPos = reconstructViewPos(uv, depth, cameraInput);
    const float3 xPos = reconstructViewPos(uv + float2(pixel.x, 0.0), dx, cameraInput);
    const float3 yPos = reconstructViewPos(uv + float2(0.0, pixel.y), dy, cameraInput);
    const float3 normal = normalize(cross(yPos - centerPos, xPos - centerPos));
    const float fresnel = pow(1.0 - saturate(normal.z), 5.0);
    const float specular = saturate(fresnel * g_FluidFresnel + g_FluidSpecularSmoothness.w * 0.25);
    const float3 color = g_FluidTint.rgb * (0.45 + 0.55 * saturate(normal.z)) +
                         g_FluidSpecularSmoothness.rgb * specular;
    const float alpha = saturate(g_FluidTint.a * (0.7 + 0.3 * fresnel));
    const float zScale = farClip / max(farClip - nearClip, 1.0e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1.0e-5);
    Out.Color = float4(color, alpha);
    Out.Depth = zScale + zTranslate / depth;
    return Out;
}
