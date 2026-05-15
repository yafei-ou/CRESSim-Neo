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
#define g_RefractionIor g_FluidCompositeMisc.y
#define g_RefractionViewThickness g_FluidCompositeMisc.z

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
Texture2DArray<float> g_FilteredFluidDepth;
SamplerState g_FilteredFluidDepth_sampler;
#if CRESSIM_FLUID_ENABLE_BACKGROUND_DISTORTION
Texture2DArray<float4> g_SceneColor;
SamplerState g_SceneColor_sampler;
#endif
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

float2 projectViewPosToUv(float3 viewPos, CameraInput cameraInput)
{
    const float aspect = max((cameraInput.viewportAndOutputSize.x * cameraInput.viewportAndOutputSize.z) /
                             max(cameraInput.viewportAndOutputSize.y * cameraInput.viewportAndOutputSize.w, 1.0e-5),
                             1.0e-5);
    const float fovRadians = max(degreesToRadians(cameraInput.projectionParams.x), degreesToRadians(1.0));
    const float tanHalfFov = tan(0.5 * fovRadians);
    const float invZ = 1.0 / max(viewPos.z, 1.0e-5);
    const float ndcX = viewPos.x * invZ / max(tanHalfFov * aspect, 1.0e-5);
    const float ndcY = viewPos.y * invZ / max(tanHalfFov, 1.0e-5);
    return float2(ndcX * 0.5 + 0.5, 0.5 - ndcY * 0.5);
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
    float3 color = g_FluidTint.rgb * (0.45 + 0.55 * saturate(normal.z)) +
                   g_FluidSpecularSmoothness.rgb * specular;
#if CRESSIM_FLUID_ENABLE_BACKGROUND_DISTORTION
    {
        const float eta = saturate(1.0 / max(g_RefractionIor, 1.0e-3));
        const float3 viewDir = normalize(centerPos);
        float3 refractedDir = refract(viewDir, normal, eta);
        if (dot(refractedDir, refractedDir) < 1.0e-6)
        {
            refractedDir = viewDir;
        }
        if (refractedDir.z <= 1.0e-4)
        {
            refractedDir.z = max(viewDir.z, 1.0e-4);
        }
        const float3 refractedSamplePos =
            centerPos + refractedDir * max(g_RefractionViewThickness, 0.0);
        const float2 offsetUv = saturate(projectViewPosToUv(refractedSamplePos, cameraInput));
        const float3 backgroundColor = g_SceneColor.SampleLevel(g_SceneColor_sampler,
                                                                float3(offsetUv, g_SceneDepthLayer), 0.0).rgb;

        color = backgroundColor * g_FluidTint.rgb * (1.0 - 0.6 * fresnel) +
                g_FluidTint.rgb * (0.15 + 0.25 * saturate(normal.z)) +
                g_FluidSpecularSmoothness.rgb * specular;
    }
#endif
    float alpha = saturate(g_FluidTint.a * (0.7 + 0.3 * fresnel));
#if CRESSIM_FLUID_ENABLE_BACKGROUND_DISTORTION
    {
        // The refracted background is the visible transmission term for the fluid surface,
        // so write the fluid result as the final surface color instead of blending it back
        // over the original undistorted scene.
        alpha = 1.0;
    }
#endif
    const float zScale = farClip / max(farClip - nearClip, 1.0e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1.0e-5);
    Out.Color = float4(color, alpha);
    Out.Depth = zScale + zTranslate / depth;
    return Out;
}
