#include "graphics/include/graphics_forward_constants.hlsli"
#include "graphics/include/graphics_scene_buffers.hlsli"

struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    nointerpolation uint CameraIndex : TEXCOORD2;
    nointerpolation uint CameraLayer : TEXCOORD3;
};

Texture2DArray g_ShadowMap0;
Texture2DArray g_ShadowMap1;
Texture2DArray g_ShadowMap2;
Texture2DArray g_ShadowMap3;
SamplerComparisonState g_ShadowMap0_sampler;
SamplerComparisonState g_ShadowMap1_sampler;
SamplerComparisonState g_ShadowMap2_sampler;
SamplerComparisonState g_ShadowMap3_sampler;

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

float ComputeShadowFactor(float3 worldPos, float3 normal, float3 lightDir, uint cameraIndex, float cameraLayer)
{
    // x: bias, y: hasShadowMap, z: minimum shadow visibility, w: reserved
    // g_MaterialParams.w controls receivesShadows per material.
    if (g_ShadowParams.y < 0.5 || g_MaterialParams.w < 0.5)
    {
        return 1.0;
    }

    PreparedCamera preparedCamera = g_PreparedCameras[cameraIndex];
    if (preparedCamera.active == 0u)
    {
        return 1.0;
    }

    float viewDepth = abs(mul(float4(worldPos, 1.0), preparedCamera.viewMatrix).z);
    float shadowDistance = preparedCamera.cascadeSplits.w;
    float fadeBand = max(preparedCamera.shadowParams.w, 1e-5);
    float distanceFade = saturate((shadowDistance - viewDepth) / fadeBand);
    if (distanceFade <= 0.0)
    {
        return 1.0;
    }

    int cascadeCount = clamp((int)round(preparedCamera.shadowParams.z), 0, 4);
    int cascadeIdx = SelectCascade(viewDepth, preparedCamera.cascadeSplits, cascadeCount);
    if (cascadeIdx < 0)
    {
        return 1.0;
    }

    float slopeScale = 1.0 - saturate(dot(normal, lightDir));
    float shadowBias = g_ShadowParams.x * (1.0 + 2.5 * slopeScale);
    float2 texelSize = max(preparedCamera.shadowParams.xy, float2(1e-5, 1e-5));
    float visibility = 1.0;

    if (cascadeIdx == 0)
    {
        visibility = SampleCascadeShadow(g_ShadowMap0, g_ShadowMap0_sampler,
                                         preparedCamera.lightViewProjectionMatrices[0], worldPos,
                                         cameraLayer, shadowBias, texelSize);
    }
    else if (cascadeIdx == 1)
    {
        visibility = SampleCascadeShadow(g_ShadowMap1, g_ShadowMap1_sampler,
                                         preparedCamera.lightViewProjectionMatrices[1], worldPos,
                                         cameraLayer, shadowBias, texelSize);
    }
    else if (cascadeIdx == 2)
    {
        visibility = SampleCascadeShadow(g_ShadowMap2, g_ShadowMap2_sampler,
                                         preparedCamera.lightViewProjectionMatrices[2], worldPos,
                                         cameraLayer, shadowBias, texelSize);
    }
    else
    {
        visibility = SampleCascadeShadow(g_ShadowMap3, g_ShadowMap3_sampler,
                                         preparedCamera.lightViewProjectionMatrices[3], worldPos,
                                         cameraLayer, shadowBias, texelSize);
    }

    float shadowTerm = lerp(g_ShadowParams.z, 1.0, visibility);
    return lerp(1.0, shadowTerm, distanceFade);
}

float4 main(in VSOutput In) : SV_Target
{
#ifdef CRESSIM_FEATURE_ALPHA_TEST
    if (g_BaseColor.w < g_MaterialParams.z)
    {
        discard;
    }
#endif

    PreparedCamera preparedCamera = g_PreparedCameras[In.CameraIndex];
    float3 N = normalize(In.WorldNormal);
    float3 V = normalize(preparedCamera.cameraPosition.xyz - In.WorldPos);
    float3 L = normalize(-g_LightDirectionIntensity.xyz);
    float3 H = normalize(V + L);

    float roughness = clamp(g_MaterialParams.y, 0.04, 1.0);
    float metallic = clamp(g_MaterialParams.x, 0.0, 1.0);
    float3 albedo = saturate(g_BaseColor.xyz);

    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, albedo, metallic);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    float3 specular = numerator / denominator;

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);
    float NdotL = max(dot(N, L), 0.0);

    float shadowFactor = ComputeShadowFactor(In.WorldPos, N, L, In.CameraIndex,
                                             (float)In.CameraLayer);

    float3 radiance = g_LightColor.xyz * g_LightDirectionIntensity.w;
    float3 diffuse = kD * albedo / PI;
    float3 Lo = (diffuse + specular) * radiance * NdotL * shadowFactor;

    float3 ambient = 0.03 * albedo;
    float3 color = ambient + Lo;
    color = color / (color + 1.0);
    color = pow(color, 1.0 / 2.2);

    return float4(color, g_BaseColor.w);
}
