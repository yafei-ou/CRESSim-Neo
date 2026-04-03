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

cbuffer GraphicsDebugParticles
{
    float4 g_DebugParticleColor;
    uint4 g_DebugParticleParams;
    float4 g_DebugParticleMisc;
};

#define g_DebugParticleCameraIndex g_DebugParticleParams.x

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);

struct PSInput
{
    float4 Position : SV_Position;
    float3 ViewPos : TEXCOORD0;
    float2 QuadCoord : TEXCOORD1;
    nointerpolation float Radius : TEXCOORD2;
};

struct PSOutput
{
    float4 Color : SV_Target;
    float Depth : SV_Depth;
};

PSOutput main(in PSInput In)
{
    PSOutput Out;

    const float quadLengthSq = dot(In.QuadCoord, In.QuadCoord);
    const float sphereTerm = 1.0 - quadLengthSq;
    if (sphereTerm < 0.0)
    {
        discard;
    }

    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_DebugParticleCameraIndex);
    const float nearClip = max(cameraInput.projectionParams.y, 1.0e-5);
    const float farClip = max(cameraInput.projectionParams.z, nearClip + 1.0e-4);

    const float3 rayDirection = normalize(In.ViewPos);
    const float t = -sqrt(sphereTerm);
    const float3 viewPos = In.ViewPos + rayDirection * (t * In.Radius);
    if (viewPos.z <= nearClip)
    {
        discard;
    }

    const float3 viewNormal = normalize(float3(In.QuadCoord, 0.0) + rayDirection * t);
    const float shade = 0.2 + 0.8 * saturate(-viewNormal.z);

    const float zScale = farClip / max(farClip - nearClip, 1.0e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1.0e-5);

    Out.Color = float4(g_DebugParticleColor.rgb * shade, g_DebugParticleColor.a);
    Out.Depth = zScale + zTranslate / viewPos.z;
    return Out;
}
