#include "include/graphics/graphics_forward_constants.hlsli"
#include "include/graphics/graphics_scene_buffers.hlsli"

struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float2 TexCoord : TEXCOORD2;
    float4 WorldTangent : TEXCOORD3;
    nointerpolation uint CameraIndex : TEXCOORD4;
    nointerpolation uint MainLightIndex : TEXCOORD5;
    nointerpolation uint ShadowLayer : TEXCOORD6;
#ifdef CRESSIM_FEATURE_DOUBLE_SIDED
    bool IsFrontFace : SV_IsFrontFace;
#endif
};

Texture2D g_BaseColorTexture;
Texture2D g_NormalTexture;
Texture2D g_MetallicRoughnessTexture;
Texture2D g_EmissiveTexture;
Texture2D g_AoTexture;
#if defined(CRESSIM_IBL_DIFFUSE_ONLY) || defined(CRESSIM_IBL_FULL)
TextureCubeArray g_IrradianceMap;
#endif
#if defined(CRESSIM_IBL_FULL)
TextureCubeArray g_PrefilteredSpecularMap;
Texture2D g_BrdfLut;
#endif
SamplerState g_BaseColorTexture_sampler;
SamplerState g_NormalTexture_sampler;
SamplerState g_MetallicRoughnessTexture_sampler;
SamplerState g_EmissiveTexture_sampler;
SamplerState g_AoTexture_sampler;
#if defined(CRESSIM_IBL_DIFFUSE_ONLY) || defined(CRESSIM_IBL_FULL)
SamplerState g_IrradianceMap_sampler;
#endif
#if defined(CRESSIM_IBL_FULL)
SamplerState g_PrefilteredSpecularMap_sampler;
SamplerState g_BrdfLut_sampler;
#endif

Texture2DArray g_ShadowMap0;
Texture2DArray g_ShadowMap1;
Texture2DArray g_ShadowMap2;
Texture2DArray g_ShadowMap3;
Texture2DArray g_LocalShadowMap;
Texture2DArray g_PointShadowMap;
SamplerComparisonState g_ShadowMap0_sampler;
SamplerComparisonState g_ShadowMap1_sampler;
SamplerComparisonState g_ShadowMap2_sampler;
SamplerComparisonState g_ShadowMap3_sampler;
SamplerComparisonState g_LocalShadowMap_sampler;
SamplerComparisonState g_PointShadowMap_sampler;

static const float PI = 3.14159265359;

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float numerator = a2;
    float denominator = (NdotH2 * (a2 - 1.0) + 1.0);
    denominator = PI * denominator * denominator;
    return numerator / max(denominator, 0.0001);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float numerator = NdotV;
    float denominator = NdotV * (1.0 - k) + k;
    return numerator / max(denominator, 0.0001);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    return F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) *
                    pow(saturate(1.0 - cosTheta), 5.0);
}

float3 BuildShadingNormal(in VSOutput In)
{
    float3 N = normalize(In.WorldNormal);

#ifdef CRESSIM_FEATURE_DOUBLE_SIDED
    const float faceSign = In.IsFrontFace ? 1.0 : -1.0;
    N *= faceSign;
#endif

#ifdef CRESSIM_FEATURE_NORMAL_MAP
    float3 T = normalize(In.WorldTangent.xyz);
#ifdef CRESSIM_FEATURE_DOUBLE_SIDED
    T *= faceSign;
#endif
    T = normalize(T - N * dot(N, T));
    float3 B = normalize(cross(N, T)) * In.WorldTangent.w;
    float3 tangentNormal = g_NormalTexture.Sample(g_NormalTexture_sampler, In.TexCoord).xyz;
    tangentNormal = tangentNormal * 2.0 - 1.0;
    float3x3 tbn = float3x3(T, B, N);
    return normalize(mul(tangentNormal, tbn));
#else
    return N;
#endif
}

int SelectCascade(float viewDepth, float4 cascadeSplits, int cascadeCount)
{
    if (cascadeCount <= 0)
    {
        return -1;
    }

    int cascadeIdx = 0;
    if (cascadeCount > 1 && viewDepth > cascadeSplits.x)
    {
        cascadeIdx = 1;
    }
    if (cascadeCount > 2 && viewDepth > cascadeSplits.y)
    {
        cascadeIdx = 2;
    }
    if (cascadeCount > 3 && viewDepth > cascadeSplits.z)
    {
        cascadeIdx = 3;
    }
    return min(cascadeIdx, cascadeCount - 1);
}

float SampleShadowPCF(Texture2D shadowMap, SamplerComparisonState shadowSampler, float2 uv, float receiverDepth, float bias, float2 texelSize)
{
    float visibility = 0.0;
    float compareDepth = receiverDepth - bias;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            float2 sampleUv = uv + float2((float)x, (float)y) * texelSize;
            if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0)
            {
                visibility += 1.0;
                continue;
            }

            visibility += shadowMap.SampleCmpLevelZero(shadowSampler, sampleUv, compareDepth);
        }
    }
    return visibility / 9.0;
}

float SampleShadowPCF(Texture2DArray shadowMap, SamplerComparisonState shadowSampler, float3 uvw,
                      float receiverDepth, float bias, float2 texelSize)
{
    float visibility = 0.0;
    float compareDepth = receiverDepth - bias;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            float2 sampleUv = uvw.xy + float2((float)x, (float)y) * texelSize;
            if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0)
            {
                visibility += 1.0;
                continue;
            }

            visibility += shadowMap.SampleCmpLevelZero(shadowSampler, float3(sampleUv, uvw.z),
                                                       compareDepth);
        }
    }
    return visibility / 9.0;
}

float SampleCascadeShadow(
    Texture2DArray shadowMap,
    SamplerComparisonState shadowSampler,
    float4x4 lightViewProjection,
    float3 worldPos,
    float layer,
    float bias,
    float2 texelSize)
{
    float4 shadowPos = mul(float4(worldPos, 1.0), lightViewProjection);
    float invW = 1.0 / max(shadowPos.w, 1e-5);
    float3 proj = shadowPos.xyz * invW;
    float2 uv = float2(0.5, 0.5) + float2(0.5, -0.5) * proj.xy;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0)
    {
        return 1.0;
    }

    return SampleShadowPCF(shadowMap, shadowSampler, float3(uv, layer), proj.z, bias, texelSize);
}

float ComputeShadowFactor(float3 worldPos, float3 normal, float3 lightDir, uint cameraIndex,
                          uint shadowLayer, LightInput mainLight)
{
    if (g_HasAnyShadowMap < 0.5 || g_MaterialParams.w < 0.5 ||
        shadowLayer == CRESSIM_INVALID_BATCH_CAMERA_LAYER)
    {
        return 1.0;
    }

    PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, cameraIndex);
    if (preparedCamera.active == 0u)
    {
        return 1.0;
    }

    float viewDepth = abs(mul(float4(worldPos, 1.0), preparedCamera.viewMatrix).z);
    float shadowDistance = preparedCamera.cascadeSplits.w;
    float fadeBand = max(preparedCamera.mainShadowFadeDistance, 1e-5);
    float distanceFade = saturate((shadowDistance - viewDepth) / fadeBand);
    if (distanceFade <= 0.0)
    {
        return 1.0;
    }

    int cascadeCount = clamp((int)round(preparedCamera.mainShadowCascadeCount), 0, 4);
    int cascadeIdx = SelectCascade(viewDepth, preparedCamera.cascadeSplits, cascadeCount);
    if (cascadeIdx < 0)
    {
        return 1.0;
    }

    float slopeScale = 1.0 - saturate(dot(normal, lightDir));
    float shadowBias = mainLight.shadowBias * (1.0 + 2.5 * slopeScale);
    float2 texelSize = max(preparedCamera.mainShadowTexelSize, float2(1e-5, 1e-5));
    float visibility = 1.0;

    if (cascadeIdx == 0)
    {
        visibility = SampleCascadeShadow(g_ShadowMap0, g_ShadowMap0_sampler,
                                         preparedCamera.lightViewProjectionMatrices[0], worldPos,
                                         (float)shadowLayer, shadowBias, texelSize);
    }
    else if (cascadeIdx == 1)
    {
        visibility = SampleCascadeShadow(g_ShadowMap1, g_ShadowMap1_sampler,
                                         preparedCamera.lightViewProjectionMatrices[1], worldPos,
                                         (float)shadowLayer, shadowBias, texelSize);
    }
    else if (cascadeIdx == 2)
    {
        visibility = SampleCascadeShadow(g_ShadowMap2, g_ShadowMap2_sampler,
                                         preparedCamera.lightViewProjectionMatrices[2], worldPos,
                                         (float)shadowLayer, shadowBias, texelSize);
    }
    else
    {
        visibility = SampleCascadeShadow(g_ShadowMap3, g_ShadowMap3_sampler,
                                         preparedCamera.lightViewProjectionMatrices[3], worldPos,
                                         (float)shadowLayer, shadowBias, texelSize);
    }

    float shadowTerm = lerp(g_ShadowMinimumVisibility, 1.0, visibility);
    return lerp(1.0, shadowTerm, distanceFade);
}

int SelectPointShadowFace(float3 dir)
{
    float3 absDir = abs(dir);
    if (absDir.x >= absDir.y && absDir.x >= absDir.z)
    {
        return dir.x >= 0.0 ? 0 : 1;
    }
    if (absDir.y >= absDir.x && absDir.y >= absDir.z)
    {
        return dir.y >= 0.0 ? 2 : 3;
    }
    return dir.z >= 0.0 ? 4 : 5;
}

float ComputeLocalShadowFactor(uint lightIndex, float3 worldPos, float3 normal, float3 lightDir)
{
    const LightShadowAssignment assignment = CRESSIM_SB_LOAD(g_LightShadowAssignments, lightIndex);
    if (assignment.shadowMode == 0u || assignment.shadowViewIndex == CRESSIM_INVALID_GPU_SCENE_INDEX)
    {
        return 1.0;
    }

    const LocalShadowView shadowView = CRESSIM_SB_LOAD(g_LocalShadowViews, assignment.shadowViewIndex);
    const LightInput light = CRESSIM_SB_LOAD(g_LightInputs, lightIndex);
    if (shadowView.active == 0u)
    {
        return 1.0;
    }

    const float2 texelSize = max(shadowView.shadowTexelSize, float2(1e-5, 1e-5));
    const float slopeScale = 1.0 - saturate(dot(normal, lightDir));
    float shadowBias = light.shadowBias;

    if (assignment.shadowMode == 1u)
    {
        // Spot-light receivers can lose shadows while still inside the lit cone if the
        // slope-scaled bias grows too quickly near the frustum edge.
        shadowBias *= (0.75 + 0.75 * slopeScale);
        return SampleCascadeShadow(g_LocalShadowMap, g_LocalShadowMap_sampler,
                                   shadowView.lightViewProjectionMatrices[0], worldPos,
                                   (float)shadowView.firstLayer, shadowBias, texelSize);
    }

    // Point-light shadows are especially sensitive to over-bias on grazing receivers like floors.
    // Keep the slope term much tighter so shadows do not disappear as they move sideways from the light.
    shadowBias *= (0.35 + 0.65 * slopeScale);

    float3 toSurface = worldPos - shadowView.lightPositionRange.xyz;
    int faceIndex = SelectPointShadowFace(toSurface);
    if (faceIndex < 0 || faceIndex >= 6)
    {
        return 1.0;
    }

    return SampleCascadeShadow(g_PointShadowMap, g_PointShadowMap_sampler,
                               shadowView.lightViewProjectionMatrices[faceIndex], worldPos,
                               (float)(shadowView.firstLayer + faceIndex), shadowBias, texelSize);
}

float3 EvaluateLocalLight(in LightInput light, float3 worldPos, float3 N, float3 V,
                          float3 albedo, float3 F0, float roughness, float metallic,
                          uint lightIndex)
{
    if (light.active == 0u || light.directionIntensity.w <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 L = float3(0.0, 0.0, 0.0);
    float attenuation = 1.0;

    if (light.type == CRESSIM_LIGHT_TYPE_DIRECTIONAL)
    {
        if (dot(light.directionIntensity.xyz, light.directionIntensity.xyz) <= 1e-6)
        {
            return float3(0.0, 0.0, 0.0);
        }
        L = normalize(-light.directionIntensity.xyz);
    }
    else
    {
        const float3 toLight = light.positionRange.xyz - worldPos;
        const float distanceSq = dot(toLight, toLight);
        const float range = max(light.positionRange.w, 1e-4);
        if (distanceSq >= range * range)
        {
            return float3(0.0, 0.0, 0.0);
        }

        const float distance = sqrt(max(distanceSq, 1e-8));
        L = toLight / max(distance, 1e-4);
        const float distance01 = saturate(distance / range);
        attenuation *= pow(1.0 - distance01, 2.0);

        if (light.type == CRESSIM_LIGHT_TYPE_SPOT)
        {
            if (dot(light.directionIntensity.xyz, light.directionIntensity.xyz) <= 1e-6)
            {
                return float3(0.0, 0.0, 0.0);
            }

            const float3 spotDir = normalize(-light.directionIntensity.xyz);
            const float cosTheta = dot(spotDir, L);
            const float outerCos = light.spotAngles.y;
            const float innerCos = max(light.spotAngles.x, outerCos + 1e-4);
            const float coneAtten = saturate((cosTheta - outerCos) / max(innerCos - outerCos, 1e-4));
            if (coneAtten <= 0.0)
            {
                return float3(0.0, 0.0, 0.0);
            }
            attenuation *= coneAtten;
        }
    }

    const float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    const float3 H = normalize(V + L);
    const float NDF = DistributionGGX(N, H, roughness);
    const float G = GeometrySmith(N, V, L, roughness);
    const float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    const float3 numerator = NDF * G * F;
    const float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    const float3 specular = numerator / denominator;
    const float3 kS = F;
    const float3 kD = (1.0 - kS) * (1.0 - metallic);
    const float3 diffuse = kD * albedo / PI;
    const float3 radiance = light.color.xyz * light.directionIntensity.w * attenuation;
    const float shadowFactor =
        light.castsShadows != 0u ? ComputeLocalShadowFactor(lightIndex, worldPos, N, L) : 1.0;
    return (diffuse + specular) * radiance * NdotL * shadowFactor;
}

float4 main(in VSOutput In) : SV_Target
{
    float4 sampledBaseColor = g_BaseColorTexture.Sample(g_BaseColorTexture_sampler, In.TexCoord);
    float4 sampledMetallicRoughness =
        g_MetallicRoughnessTexture.Sample(g_MetallicRoughnessTexture_sampler, In.TexCoord);
    float3 sampledEmissive = g_EmissiveTexture.Sample(g_EmissiveTexture_sampler, In.TexCoord).xyz;
    float ao = g_AoTexture.Sample(g_AoTexture_sampler, In.TexCoord).x;

    float4 baseColor = float4(g_BaseColorFactor.rgb * sampledBaseColor.rgb,
                              g_BaseColorFactor.a * sampledBaseColor.a);
    float roughness = clamp(g_MaterialParams.y * sampledMetallicRoughness.g, 0.04, 1.0);
    float metallic = clamp(g_MaterialParams.x * sampledMetallicRoughness.b, 0.0, 1.0);
    float3 emissive = g_EmissiveFactor.rgb * sampledEmissive;

#ifdef CRESSIM_FEATURE_ALPHA_TEST
    if (baseColor.w < g_MaterialParams.z)
    {
        discard;
    }
#endif

    PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, In.CameraIndex);
    float3 N = BuildShadingNormal(In);
    float3 V = normalize(preparedCamera.cameraPosition.xyz - In.WorldPos);

    float3 albedo = saturate(baseColor.xyz);

    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, albedo, metallic);

    float3 Lo = float3(0.0, 0.0, 0.0);
    if (In.MainLightIndex != CRESSIM_INVALID_GPU_SCENE_INDEX)
    {
        const LightInput mainLight = CRESSIM_SB_LOAD(g_LightInputs, In.MainLightIndex);
        const bool hasMainLight =
            mainLight.active != 0u &&
            dot(mainLight.directionIntensity.xyz, mainLight.directionIntensity.xyz) > 1e-6 &&
            mainLight.directionIntensity.w > 0.0;
        if (hasMainLight)
        {
            float3 L = normalize(-mainLight.directionIntensity.xyz);
            float3 H = normalize(V + L);
            float NDF = DistributionGGX(N, H, roughness);
            float G = GeometrySmith(N, V, L, roughness);
            float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

            float3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
            float3 specular = numerator / denominator;

            float3 kS = F;
            float3 kD = (1.0 - kS) * (1.0 - metallic);
            float NdotL = max(dot(N, L), 0.0);
            float shadowFactor =
                ComputeShadowFactor(In.WorldPos, N, L, In.CameraIndex, In.ShadowLayer, mainLight);
            float3 radiance = mainLight.color.xyz * mainLight.directionIntensity.w;
            float3 diffuse = kD * albedo / PI;
            Lo = (diffuse + specular) * radiance * NdotL * shadowFactor;
        }
    }

    const LocalLightSelection localSelection = CRESSIM_SB_LOAD(g_LocalLightSelections, preparedCamera.envIndex);
    [unroll]
    for (uint localLightIdx = 0u; localLightIdx < 8u; ++localLightIdx)
    {
        if (localLightIdx >= localSelection.localLightCount)
        {
            continue;
        }

        const uint lightIndex = localSelection.lightIndices[localLightIdx];
        const LightInput localLight = CRESSIM_SB_LOAD(g_LightInputs, lightIndex);
        Lo += EvaluateLocalLight(localLight, In.WorldPos, N, V, albedo, F0, roughness, metallic,
                                 lightIndex);
    }

    float NdotV = max(dot(N, V), 0.0);
    float3 F = FresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float3 ambient = 0.03 * albedo * ao;
#if defined(CRESSIM_IBL_DIFFUSE_ONLY) || defined(CRESSIM_IBL_FULL)
    EnvironmentIblLookupEntry iblEntry = CRESSIM_SB_LOAD(g_EnvironmentIblLookup, preparedCamera.envIndex);
    if (iblEntry.enabled != 0u)
    {
        float3 irradiance =
            g_IrradianceMap.Sample(g_IrradianceMap_sampler, float4(N, (float)iblEntry.sliceIndex))
                .rgb;
        float3 diffuseIbl = irradiance * albedo;

        float iblStrength = step(1.0e-5, dot(irradiance, irradiance));
#if defined(CRESSIM_IBL_FULL)
        float3 iblAmbient = kD * diffuseIbl;
        float3 R = reflect(-V, N);
        float mipLevel = roughness * max(g_IblSpecularParams.x - 1.0, 0.0);
        float3 prefilteredColor =
            g_PrefilteredSpecularMap.SampleLevel(g_PrefilteredSpecularMap_sampler,
                                                 float4(R, (float)iblEntry.sliceIndex), mipLevel)
                .rgb;
        float2 brdf = g_BrdfLut.Sample(g_BrdfLut_sampler, float2(NdotV, roughness)).rg;
        float3 specularIbl = prefilteredColor * (F * brdf.x + brdf.y);
        iblAmbient += specularIbl;
        iblStrength =
            step(1.0e-5, dot(irradiance, irradiance) + dot(prefilteredColor, prefilteredColor));
#else
        // Diffuse-only IBL is an ambient-fill mode, so avoid the Fresnel energy split that
        // assumes a matching specular IBL term exists at grazing angles.
        float3 iblAmbient = diffuseIbl * (1.0 - metallic);
#endif

        iblAmbient *= ao * iblEntry.intensity;
        ambient = lerp(ambient, iblAmbient, iblStrength);
    }
#endif
    float3 color = ambient + Lo + emissive;

    return float4(color, baseColor.w);
}
