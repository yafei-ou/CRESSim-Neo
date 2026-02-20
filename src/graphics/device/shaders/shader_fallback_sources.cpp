#include "graphics/device/shaders/shader_fallback_sources.h"

namespace cressim::neo::graphics::shaders
{

namespace
{

constexpr char kPbrVsSource[] = R"(
struct VSInput
{
    float3 Position : ATTRIB0;
    float3 Normal : ATTRIB1;
    float2 TexCoord : ATTRIB2;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
};

cbuffer PbrConstants
{
    float4x4 g_Model;
    float4x4 g_ViewMatrix;
    float4x4 g_ViewProjection;
    float4x4 g_LightViewProjection[4];
    float4x4 g_NormalMatrix;
    float4 g_CameraPositionMetallic;
    float4 g_LightDirectionIntensity;
    float4 g_LightColorRoughness;
    float4 g_BaseColor;
    float4 g_CascadeSplits;
    float4 g_ShadowTexelSizeCascadeCount;
    float4 g_ShadowParams;
    float4 g_PipelineParams;
};

void main(in VSInput In, out VSOutput Out)
{
    float4 worldPos = mul(float4(In.Position, 1.0), g_Model);
    Out.Position = mul(worldPos, g_ViewProjection);
    Out.WorldPos = worldPos.xyz;
    Out.WorldNormal = normalize(mul(float4(In.Normal, 0.0), g_NormalMatrix).xyz);
}
)";

constexpr char kPbrPsSource[] = R"(
struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
};

cbuffer PbrConstants
{
    float4x4 g_Model;
    float4x4 g_ViewMatrix;
    float4x4 g_ViewProjection;
    float4x4 g_LightViewProjection[4];
    float4x4 g_NormalMatrix;
    float4 g_CameraPositionMetallic;
    float4 g_LightDirectionIntensity;
    float4 g_LightColorRoughness;
    float4 g_BaseColor;
    float4 g_CascadeSplits;
    float4 g_ShadowTexelSizeCascadeCount;
    float4 g_ShadowParams;
    float4 g_PipelineParams;
};

Texture2D g_ShadowMap0;
Texture2D g_ShadowMap1;
Texture2D g_ShadowMap2;
Texture2D g_ShadowMap3;
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

int SelectCascade(float viewDepth)
{
    int cascadeCount = clamp((int)round(g_ShadowTexelSizeCascadeCount.z), 0, 4);
    if (cascadeCount <= 0)
    {
        return -1;
    }

    int cascadeIdx = 0;
    if (cascadeCount > 1 && viewDepth > g_CascadeSplits.x)
    {
        cascadeIdx = 1;
    }
    if (cascadeCount > 2 && viewDepth > g_CascadeSplits.y)
    {
        cascadeIdx = 2;
    }
    if (cascadeCount > 3 && viewDepth > g_CascadeSplits.z)
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

float SampleCascadeShadow(
    Texture2D shadowMap,
    SamplerComparisonState shadowSampler,
    float4x4 lightViewProjection,
    float3 worldPos,
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

    return SampleShadowPCF(shadowMap, shadowSampler, uv, proj.z, bias, texelSize);
}

float ComputeShadowFactor(float3 worldPos, float3 normal, float3 lightDir)
{
    // x: bias, y: hasShadowMap, z: receivesShadows, w: minimum shadow visibility
    if (g_ShadowParams.y < 0.5 || g_ShadowParams.z < 0.5)
    {
        return 1.0;
    }

    float viewDepth = mul(float4(worldPos, 1.0), g_ViewMatrix).z;
    float shadowDistance = g_CascadeSplits.w;
    float fadeBand = max(g_ShadowTexelSizeCascadeCount.w, 1e-5);
    float distanceFade = saturate((shadowDistance - viewDepth) / fadeBand);
    if (distanceFade <= 0.0)
    {
        return 1.0;
    }

    int cascadeIdx = SelectCascade(viewDepth);
    if (cascadeIdx < 0)
    {
        return 1.0;
    }

    float slopeScale = 1.0 - saturate(dot(normal, lightDir));
    float shadowBias = g_ShadowParams.x * (1.0 + 2.5 * slopeScale);
    float2 texelSize = max(g_ShadowTexelSizeCascadeCount.xy, float2(1e-5, 1e-5));
    float visibility = 1.0;

    if (cascadeIdx == 0)
    {
        visibility = SampleCascadeShadow(g_ShadowMap0, g_ShadowMap0_sampler, g_LightViewProjection[0], worldPos, shadowBias, texelSize);
    }
    else if (cascadeIdx == 1)
    {
        visibility = SampleCascadeShadow(g_ShadowMap1, g_ShadowMap1_sampler, g_LightViewProjection[1], worldPos, shadowBias, texelSize);
    }
    else if (cascadeIdx == 2)
    {
        visibility = SampleCascadeShadow(g_ShadowMap2, g_ShadowMap2_sampler, g_LightViewProjection[2], worldPos, shadowBias, texelSize);
    }
    else
    {
        visibility = SampleCascadeShadow(g_ShadowMap3, g_ShadowMap3_sampler, g_LightViewProjection[3], worldPos, shadowBias, texelSize);
    }

    float shadowTerm = lerp(g_ShadowParams.w, 1.0, visibility);
    return lerp(1.0, shadowTerm, distanceFade);
}

float4 main(in VSOutput In) : SV_Target
{
#ifdef CRESSIM_FEATURE_ALPHA_TEST
    if (g_BaseColor.w < g_PipelineParams.x)
    {
        discard;
    }
#endif

    float3 N = normalize(In.WorldNormal);
    float3 V = normalize(g_CameraPositionMetallic.xyz - In.WorldPos);
    float3 L = normalize(-g_LightDirectionIntensity.xyz);
    float3 H = normalize(V + L);

    float roughness = clamp(g_LightColorRoughness.w, 0.04, 1.0);
    float metallic = clamp(g_CameraPositionMetallic.w, 0.0, 1.0);
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

    float shadowFactor = ComputeShadowFactor(In.WorldPos, N, L);

    float3 radiance = g_LightColorRoughness.xyz * g_LightDirectionIntensity.w;
    float3 diffuse = kD * albedo / PI;
    float3 Lo = (diffuse + specular) * radiance * NdotL * shadowFactor;

    float3 ambient = 0.03 * albedo;
    float3 color = ambient + Lo;
    color = color / (color + 1.0);
    color = pow(color, 1.0 / 2.2);

    return float4(color, g_BaseColor.w);
}
)";

constexpr char kShadowDepthVsSource[] = R"(
struct VSInput
{
    float3 Position : ATTRIB0;
    float3 Normal : ATTRIB1;
    float2 TexCoord : ATTRIB2;
};

struct VSOutput
{
    float4 Position : SV_Position;
};

cbuffer ShadowConstants
{
    float4x4 g_Model;
    float4x4 g_LightViewProjection;
};

void main(in VSInput In, out VSOutput Out)
{
    float4 worldPos = mul(float4(In.Position, 1.0), g_Model);
    Out.Position = mul(worldPos, g_LightViewProjection);
}
)";

} // namespace

const char* pbrVertex()
{
    return kPbrVsSource;
}

const char* pbrPixel()
{
    return kPbrPsSource;
}

const char* shadowDepthVertex()
{
    return kShadowDepthVsSource;
}

} // namespace cressim::neo::graphics::shaders
