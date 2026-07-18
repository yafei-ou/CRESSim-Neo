#include "structured_buffer_compat.hlsli"
#include "graphics_camera_input.hlsli"

cbuffer GraphicsFluidDepth
{
    uint4 g_FluidDepthParams;
    float4 g_FluidDepthMisc;
};

#define g_FluidDepthCameraIndex g_FluidDepthParams.x
#define g_FluidDepthDepthEdgeThreshold g_FluidDepthMisc.y

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
#ifndef CRESSIM_FLUID_SCENE_INPUTS_ARRAY
#    define CRESSIM_FLUID_SCENE_INPUTS_ARRAY 1
#endif

#if CRESSIM_FLUID_SCENE_INPUTS_ARRAY
Texture2DArray<float> g_SceneDepth;
#else
Texture2D<float> g_SceneDepth;
#endif
SamplerState g_SceneDepth_sampler;

struct PSInput
{
    float4 Position : SV_Position;
    float3 ViewPos : TEXCOORD0;
    float2 Uv : TEXCOORD1;
    float4 Ani1 : TEXCOORD2;
    float4 Ani2 : TEXCOORD3;
    float4 Ani3 : TEXCOORD4;
    float AniScale : TEXCOORD5;
    nointerpolation uint SceneDepthLayer : TEXCOORD6;
};

float linearizeDepth(float depth, float nearClip, float farClip)
{
    const float zScale = farClip / max(farClip - nearClip, 1.0e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1.0e-5);
    return zTranslate / max(depth - zScale, -1.0e-5);
}

float main(PSInput In) : SV_Target
{
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_FluidDepthCameraIndex);
    const float nearClip = max(cameraInput.projectionParams.y, 1.0e-4);
    const float farClip = max(cameraInput.projectionParams.z, nearClip + 1.0e-4);

    const float3x3 aniTrans = transpose(float3x3(In.Ani1.xyz * In.Ani1.w,
                                                 In.Ani2.xyz * In.Ani2.w,
                                                 In.Ani3.xyz * In.Ani3.w));
    const float3x3 invAniTrans = float3x3(In.Ani1.xyz / In.Ani1.w,
                                          In.Ani2.xyz / In.Ani2.w,
                                          In.Ani3.xyz / In.Ani3.w);

    float3 quadCoord = float3(In.Uv * 2.0 - 1.0, 0.0);
    float3 rayDirection = normalize(In.ViewPos);

    quadCoord = mul(invAniTrans, quadCoord);
    rayDirection = mul(invAniTrans, rayDirection);

    const float a = dot(rayDirection, rayDirection);
    const float halfB = dot(rayDirection, quadCoord);
    const float c = dot(quadCoord, quadCoord) - 1.0;
    const float delta = halfB * halfB - a * c;
    if (a <= 1.0e-6 || delta < 0.0)
    {
        discard;
    }

    const float t = (-halfB - sqrt(delta)) / a;
    const float3 viewPos = In.ViewPos + mul(aniTrans, rayDirection * t) * In.AniScale;
    const float linearDepth = viewPos.z;
    if (linearDepth <= nearClip || linearDepth >= farClip)
    {
        discard;
    }

    const float2 uv = In.Position.xy / cameraInput.viewportAndOutputSize.zw;
#if CRESSIM_FLUID_SCENE_INPUTS_ARRAY
    const float sceneDepth = g_SceneDepth.SampleLevel(g_SceneDepth_sampler,
                                                      float3(uv, In.SceneDepthLayer), 0.0);
#else
    const float sceneDepth = g_SceneDepth.SampleLevel(g_SceneDepth_sampler, uv, 0.0);
#endif
    const float linearSceneDepth = linearizeDepth(sceneDepth, nearClip, farClip);
    if (linearDepth >= linearSceneDepth - g_FluidDepthDepthEdgeThreshold)
    {
        discard;
    }

    return linearDepth;
}
