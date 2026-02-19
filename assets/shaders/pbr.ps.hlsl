struct VSOutput
{
    float4 Position : SV_Position;
    float3 WorldPos : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float4 ShadowPos : TEXCOORD2;
};

cbuffer PbrConstants
{
    row_major float4x4 g_Model;
    row_major float4x4 g_ViewProjection;
    row_major float4x4 g_LightViewProjection;
    float4 g_CameraPositionMetallic;
    float4 g_LightDirectionIntensity;
    float4 g_LightColorRoughness;
    float4 g_BaseColor;
    float4 g_ShadowParams;
};

Texture2D g_ShadowMap;
SamplerState g_ShadowMap_sampler;

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

float ComputeShadowFactor(float4 shadowPos)
{
    // x: bias, y: hasShadowMap, z: receivesShadows
    if (g_ShadowParams.y < 0.5 || g_ShadowParams.z < 0.5)
    {
        return 1.0;
    }

    float invW = 1.0 / max(shadowPos.w, 1e-5);
    float3 proj = shadowPos.xyz * invW;
    float2 uv = float2(0.5, 0.5) + float2(0.5, -0.5) * proj.xy;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0)
    {
        return 1.0;
    }

    float mapDepth = g_ShadowMap.SampleLevel(g_ShadowMap_sampler, uv, 0).r;
    float lit = (proj.z - g_ShadowParams.x) <= mapDepth ? 1.0 : 0.35;
    return lit;
}

float4 main(in VSOutput In) : SV_Target
{
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

    float shadowFactor = ComputeShadowFactor(In.ShadowPos);

    float3 radiance = g_LightColorRoughness.xyz * g_LightDirectionIntensity.w;
    float3 diffuse = kD * albedo / PI;
    float3 Lo = (diffuse + specular) * radiance * NdotL * shadowFactor;

    float3 ambient = 0.03 * albedo;
    float3 color = ambient + Lo;
    color = color / (color + 1.0);
    color = pow(color, 1.0 / 2.2);

    return float4(color, g_BaseColor.w);
}
