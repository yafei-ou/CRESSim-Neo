#include "include/structured_buffer_compat.hlsli"
#include "include/graphics/graphics_camera_input.hlsli"

static const uint CRESSIM_LIGHT_TYPE_DIRECTIONAL = 0u;
static const uint CRESSIM_LIGHT_TYPE_POINT = 1u;
static const uint CRESSIM_LIGHT_TYPE_SPOT = 2u;
static const uint CRESSIM_INVALID_GPU_SCENE_INDEX = 0xffffffffu;
static const uint CRESSIM_FORWARD_LOCAL_LIGHT_CAP = 8u;
static const float PI = 3.14159265359;

struct LightInput
{
    float4 positionRange;
    float4 directionIntensity;
    float4 color;
    float4 spotAngles;
    float shadowDistance;
    float shadowFadeDistance;
    float shadowBias;
    float shadowPadding0;
    uint envIndex;
    uint lightSlot;
    uint type;
    uint active;
    uint castsShadows;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct LocalLightSelection
{
    uint localLightCount;
    uint shadowedLocalLightCount;
    uint shadowedPointLightCount;
    uint reserved0;
    uint lightIndices[CRESSIM_FORWARD_LOCAL_LIGHT_CAP];
};

cbuffer GraphicsFluidComposite
{
    float4 g_FluidSpecularSmoothness;
    uint4 g_FluidCompositeParams;
    float4 g_FluidCompositeMisc;
    float4 g_FluidCompositeMisc2;
    float4 g_FluidViewportRect;
};

#define g_FluidCameraIndex g_FluidCompositeParams.x
#define g_FluidDepthLayer g_FluidCompositeParams.y
#define g_SceneDepthLayer g_FluidCompositeParams.z
#define g_MainLightIndex g_FluidCompositeParams.w
#define g_FluidFresnel g_FluidCompositeMisc.x
#define g_RefractionIor g_FluidCompositeMisc.y
#define g_RefractionViewThickness g_FluidCompositeMisc.z
#define g_NormalReconstructionDepthThreshold g_FluidCompositeMisc2.x

#ifndef CRESSIM_FLUID_SCENE_INPUTS_ARRAY
#    define CRESSIM_FLUID_SCENE_INPUTS_ARRAY 1
#endif

float2 localUvToFullUv(float2 localUv)
{
    return g_FluidViewportRect.xy + localUv * g_FluidViewportRect.zw;
}

float2 fullUvToLocalUv(float2 fullUv)
{
    const float2 viewportSize = max(g_FluidViewportRect.zw, float2(1.0e-5, 1.0e-5));
    return (fullUv - g_FluidViewportRect.xy) / viewportSize;
}

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_STRUCTURED_BUFFER(LightInput, g_LightInputs);
CRESSIM_STRUCTURED_BUFFER(LocalLightSelection, g_LocalLightSelections);
Texture2DArray<float> g_FilteredFluidDepth;
SamplerState g_FilteredFluidDepth_sampler;
Texture2DArray<float4> g_FluidSurfaceColor;
SamplerState g_FluidSurfaceColor_sampler;
#if CRESSIM_FLUID_ENABLE_BACKGROUND_REFRACTION
#    if CRESSIM_FLUID_SCENE_INPUTS_ARRAY
Texture2DArray<float4> g_SceneColor;
#    else
Texture2D<float4> g_SceneColor;
#    endif
SamplerState g_SceneColor_sampler;
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

float3 safeNormalize(float3 v, float3 fallback)
{
    const float lenSq = dot(v, v);
    if (lenSq <= 1.0e-8)
    {
        return fallback;
    }
    return v * rsqrt(lenSq);
}

float3 safeNormalizeWithMinLenSq(float3 v, float3 fallback, float minLenSq)
{
    const float lenSq = dot(v, v);
    if (lenSq <= minLenSq)
    {
        return fallback;
    }
    return v * rsqrt(lenSq);
}

float3 quaternionRotateVector(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float NdotH = max(dot(N, H), 0.0);
    const float NdotH2 = NdotH * NdotH;
    const float numerator = a2;
    float denominator = (NdotH2 * (a2 - 1.0) + 1.0);
    denominator = PI * denominator * denominator;
    return numerator / max(denominator, 1.0e-4);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    const float r = roughness + 1.0;
    const float k = (r * r) / 8.0;
    const float numerator = NdotV;
    const float denominator = NdotV * (1.0 - k) + k;
    return numerator / max(denominator, 1.0e-4);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    const float NdotV = max(dot(N, V), 0.0);
    const float NdotL = max(dot(N, L), 0.0);
    const float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    const float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float3 EvaluateFluidLight(in LightInput light, float3 worldPos, float3 N, float3 V, float3 albedo,
                          float3 surfaceSpecularColor, float roughness, float fluidOpacity)
{
    if (light.active == 0u || light.directionIntensity.w <= 0.0)
    {
        return 0.0;
    }

    float3 L = 0.0;
    float attenuation = 1.0;

    if (light.type == CRESSIM_LIGHT_TYPE_DIRECTIONAL)
    {
        if (dot(light.directionIntensity.xyz, light.directionIntensity.xyz) <= 1.0e-6)
        {
            return 0.0;
        }
        L = normalize(-light.directionIntensity.xyz);
    }
    else
    {
        const float3 toLight = light.positionRange.xyz - worldPos;
        const float distanceSq = dot(toLight, toLight);
        const float range = max(light.positionRange.w, 1.0e-4);
        if (distanceSq >= range * range)
        {
            return 0.0;
        }

        const float distance = sqrt(max(distanceSq, 1.0e-8));
        L = toLight / max(distance, 1.0e-4);
        const float distance01 = saturate(distance / range);
        attenuation *= pow(1.0 - distance01, 2.0);

        if (light.type == CRESSIM_LIGHT_TYPE_SPOT)
        {
            if (dot(light.directionIntensity.xyz, light.directionIntensity.xyz) <= 1.0e-6)
            {
                return 0.0;
            }

            const float3 spotDir = normalize(-light.directionIntensity.xyz);
            const float cosTheta = dot(spotDir, L);
            const float outerCos = light.spotAngles.y;
            const float innerCos = max(light.spotAngles.x, outerCos + 1.0e-4);
            const float coneAtten =
                saturate((cosTheta - outerCos) / max(innerCos - outerCos, 1.0e-4));
            if (coneAtten <= 0.0)
            {
                return 0.0;
            }
            attenuation *= coneAtten;
        }
    }

    const float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0)
    {
        return 0.0;
    }

    const float3 H = normalize(V + L);
    const float3 F = FresnelSchlick(max(dot(H, V), 0.0), surfaceSpecularColor);
    const float NDF = DistributionGGX(N, H, roughness);
    const float G = GeometrySmith(N, V, L, roughness);
    const float3 numerator = NDF * G * F;
    const float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 1.0e-4;
    const float3 specular = numerator / denominator;
    const float3 diffuse = (1.0 - F) * albedo * fluidOpacity / PI;
    const float3 radiance = light.color.xyz * light.directionIntensity.w * attenuation;
    return (diffuse + specular * fluidOpacity) * radiance * NdotL;
}

float3 reconstructNeighborViewPos(float2 fullUv, float sampleDepth, float3 centerPos,
                                  CameraInput cameraInput, float threshold)
{
    if (sampleDepth > 999999.0 || abs(sampleDepth - centerPos.z) > threshold)
    {
        return centerPos;
    }
    return reconstructViewPos(fullUvToLocalUv(fullUv), sampleDepth, cameraInput);
}

PSOutput main(PSInput In)
{
    PSOutput Out;
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_FluidCameraIndex);
    const float2 localUv = saturate(In.TexCoord);
    const float2 fullUv = saturate(localUvToFullUv(localUv));
    const float depth = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                         float3(fullUv, g_FluidDepthLayer), 0.0);
    if (depth > 999999.0)
    {
        discard;
    }

    const float nearClip = max(cameraInput.projectionParams.y, 1.0e-4);
    const float farClip = max(cameraInput.projectionParams.z, nearClip + 1.0e-4);
#if CRESSIM_FLUID_SCENE_INPUTS_ARRAY
    const float sceneDepth = g_SceneDepth.SampleLevel(g_SceneDepth_sampler,
                                                      float3(fullUv, g_SceneDepthLayer), 0.0);
#else
    const float sceneDepth = g_SceneDepth.SampleLevel(g_SceneDepth_sampler, fullUv, 0.0);
#endif
    const float linearSceneDepth = linearizeDepth(sceneDepth, nearClip, farClip);
    if (depth >= linearSceneDepth)
    {
        discard;
    }

    const float2 pixel = float2(1.0 / max(cameraInput.viewportAndOutputSize.z, 1.0),
                                1.0 / max(cameraInput.viewportAndOutputSize.w, 1.0));
    const float dx1 = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                       float3(fullUv + float2(pixel.x, 0.0), g_FluidDepthLayer), 0.0);
    const float dx2 = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                       float3(fullUv - float2(pixel.x, 0.0), g_FluidDepthLayer), 0.0);
    const float dy1 = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                       float3(fullUv + float2(0.0, pixel.y), g_FluidDepthLayer), 0.0);
    const float dy2 = g_FilteredFluidDepth.SampleLevel(g_FilteredFluidDepth_sampler,
                                                       float3(fullUv - float2(0.0, pixel.y), g_FluidDepthLayer), 0.0);
    const float3 centerPos = reconstructViewPos(localUv, depth, cameraInput);
    const float normalDepthThreshold = max(g_NormalReconstructionDepthThreshold, 1.0e-4);
    const float3 xPos1 = reconstructNeighborViewPos(fullUv + float2(pixel.x, 0.0), dx1, centerPos,
                                                    cameraInput, normalDepthThreshold);
    const float3 xPos2 = reconstructNeighborViewPos(fullUv - float2(pixel.x, 0.0), dx2, centerPos,
                                                    cameraInput, normalDepthThreshold);
    const float3 yPos1 = reconstructNeighborViewPos(fullUv + float2(0.0, pixel.y), dy1, centerPos,
                                                    cameraInput, normalDepthThreshold);
    const float3 yPos2 = reconstructNeighborViewPos(fullUv - float2(0.0, pixel.y), dy2, centerPos,
                                                    cameraInput, normalDepthThreshold);
    const float3 xForward = xPos1 - centerPos;
    const float3 xBackward = centerPos - xPos2;
    const float3 yForward = yPos1 - centerPos;
    const float3 yBackward = centerPos - yPos2;
    const float3 xTangent = (dot(xForward, xForward) >= dot(xBackward, xBackward)) ? xForward : xBackward;
    const float3 yTangent = (dot(yForward, yForward) >= dot(yBackward, yBackward)) ? yForward : yBackward;
    const float tangentMinLenSq = 1.0e-20;
    const float3 xCentral = xPos1 - xPos2;
    const float3 yCentral = yPos1 - yPos2;
    float3 normal = float3(0.0, 0.0, 0.0);
    if (dot(xCentral, xCentral) > tangentMinLenSq &&
        dot(yCentral, yCentral) > tangentMinLenSq)
    {
        const float3 xDir =
            safeNormalizeWithMinLenSq(xCentral, float3(1.0, 0.0, 0.0), tangentMinLenSq);
        const float3 yDir =
            safeNormalizeWithMinLenSq(yCentral, float3(0.0, 1.0, 0.0), tangentMinLenSq);
        normal = cross(xDir, yDir);
    }
    if (dot(normal, normal) <= 1.0e-12 &&
        dot(xTangent, xTangent) > tangentMinLenSq &&
        dot(yTangent, yTangent) > tangentMinLenSq)
    {
        const float3 xDir =
            safeNormalizeWithMinLenSq(xTangent, float3(1.0, 0.0, 0.0), tangentMinLenSq);
        const float3 yDir =
            safeNormalizeWithMinLenSq(yTangent, float3(0.0, 1.0, 0.0), tangentMinLenSq);
        normal = cross(xDir, yDir);
    }
    normal = safeNormalizeWithMinLenSq(normal, float3(0.0, 0.0, 1.0), 1.0e-12);
    const float3 viewDir = safeNormalize(centerPos, float3(0.0, 0.0, 1.0));
    if (dot(normal, viewDir) > 0.0)
    {
        normal = -normal;
    }
    const float ndotv = saturate(dot(normal, -viewDir));
    const float fresnel = pow(1.0 - ndotv, 5.0);
    const float surfaceFresnel = 0.6 * lerp(saturate(g_FluidFresnel), 1.0, fresnel);
    const float roughness = clamp(1.0 - g_FluidSpecularSmoothness.w, 0.04, 1.0);
    const float4 surfaceColor = g_FluidSurfaceColor.SampleLevel(
        g_FluidSurfaceColor_sampler, float3(fullUv, g_FluidDepthLayer), 0.0);
    const float3 fluidColor = saturate(surfaceColor.rgb);
    const float fluidOpacity = saturate(surfaceColor.a);
    const float transmissionStrength = (1.0 - fluidOpacity) * (1.0 - surfaceFresnel);
    const float3 worldNormal =
        safeNormalize(quaternionRotateVector(cameraInput.orientation, normal), float3(0.0, 1.0, 0.0));
    const float3 worldViewDir =
        safeNormalize(-quaternionRotateVector(cameraInput.orientation, viewDir), float3(0.0, 0.0, 1.0));
    const float3 worldPos =
        quaternionRotateVector(cameraInput.orientation, centerPos) + cameraInput.position.xyz;
    const float3 surfaceSpecularColor = saturate(g_FluidSpecularSmoothness.rgb * surfaceFresnel);

    float3 transmissionColor = 0.0;
#if CRESSIM_FLUID_ENABLE_BACKGROUND_REFRACTION
    {
        const float eta = saturate(1.0 / max(g_RefractionIor, 1.0e-3));
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
        const float2 offsetLocalUv = saturate(projectViewPosToUv(refractedSamplePos, cameraInput));
        const float2 offsetUv = saturate(localUvToFullUv(offsetLocalUv));
#if CRESSIM_FLUID_SCENE_INPUTS_ARRAY
        const float3 refractedBackgroundColor =
            g_SceneColor.SampleLevel(g_SceneColor_sampler, float3(offsetUv, g_SceneDepthLayer), 0.0).rgb;
#else
        const float3 refractedBackgroundColor =
            g_SceneColor.SampleLevel(g_SceneColor_sampler, offsetUv, 0.0).rgb;
#endif
        const float3 transmissionTint =
            lerp(float3(1.0, 1.0, 1.0), fluidColor, 0.35 * fluidOpacity);
        transmissionColor = refractedBackgroundColor * transmissionTint * transmissionStrength;
    }
#endif

    float3 directLighting = 0.0;
    if (g_MainLightIndex != CRESSIM_INVALID_GPU_SCENE_INDEX)
    {
        const LightInput mainLight = CRESSIM_SB_LOAD(g_LightInputs, g_MainLightIndex);
        directLighting += EvaluateFluidLight(mainLight, worldPos, worldNormal, worldViewDir,
                                             fluidColor, surfaceSpecularColor, roughness,
                                             fluidOpacity);
    }

    const LocalLightSelection localSelection =
        CRESSIM_SB_LOAD(g_LocalLightSelections, cameraInput.envIndex);
    [unroll]
    for (uint localLightIdx = 0u; localLightIdx < CRESSIM_FORWARD_LOCAL_LIGHT_CAP; ++localLightIdx)
    {
        if (localLightIdx >= localSelection.localLightCount)
        {
            continue;
        }

        const uint lightIndex = localSelection.lightIndices[localLightIdx];
        const LightInput localLight = CRESSIM_SB_LOAD(g_LightInputs, lightIndex);
        directLighting += EvaluateFluidLight(localLight, worldPos, worldNormal, worldViewDir,
                                             fluidColor, surfaceSpecularColor, roughness,
                                             fluidOpacity);
    }

    const float3 ambient = fluidColor * fluidOpacity * (0.03 + 0.05 * ndotv);
    float3 color = transmissionColor + ambient + directLighting;

    float alpha = fluidOpacity;
#if CRESSIM_FLUID_ENABLE_BACKGROUND_REFRACTION
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
