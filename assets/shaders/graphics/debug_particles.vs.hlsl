#include "graphics/include/graphics_scene_buffers.hlsli"

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
#define g_DebugParticleTargetLayer g_DebugParticleParams.y
#define g_DebugParticleEnvIndex g_DebugParticleParams.z
#define g_DebugParticleFlags g_DebugParticleParams.w
#define g_DebugParticleFallbackRadius g_DebugParticleMisc.x
#define CRESSIM_DEBUG_PARTICLE_USE_RADII 1u

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleEnvironmentIndices);

struct VSOutput
{
    float4 Position : SV_Position;
    float3 ViewPos : TEXCOORD0;
    float2 QuadCoord : TEXCOORD1;
    nointerpolation float Radius : TEXCOORD2;
#if MANUAL_LAYER_EXPORT
    uint Layer : SV_RenderTargetArrayIndex;
#endif
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

float4x4 buildProjectionMatrix(float verticalFovDegrees, float aspect, float nearClip, float farClip)
{
    const float fovRadians = max(degreesToRadians(verticalFovDegrees), degreesToRadians(1.0));
    const float yScale = 1.0 / tan(0.5 * fovRadians);
    const float xScale = yScale / max(aspect, 1e-5);
    const float zScale = farClip / max(farClip - nearClip, 1e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1e-5);

    return float4x4(
        xScale, 0.0,    0.0,       0.0,
        0.0,    yScale, 0.0,       0.0,
        0.0,    0.0,    zScale,    1.0,
        0.0,    0.0,    zTranslate, 0.0);
}

float2 quadCornerForVertex(uint triangleVertexIndex)
{
    if (triangleVertexIndex == 0u)
    {
        return float2(-1.0, -1.0);
    }
    if (triangleVertexIndex == 1u)
    {
        return float2(1.0, -1.0);
    }
    if (triangleVertexIndex == 2u)
    {
        return float2(-1.0, 1.0);
    }
    if (triangleVertexIndex == 3u)
    {
        return float2(-1.0, 1.0);
    }
    if (triangleVertexIndex == 4u)
    {
        return float2(1.0, -1.0);
    }
    return float2(1.0, 1.0);
}

void main(uint vertexId : SV_VertexID, out VSOutput Out)
{
    const uint particleIndex = vertexId / 6u;
    const uint triangleVertexIndex = vertexId % 6u;
    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, g_DebugParticleCameraIndex);
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_DebugParticleCameraIndex);

    const float4 particlePositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const uint particleEnvIndex = CRESSIM_SB_LOAD(g_ParticleEnvironmentIndices, particleIndex);
    float particleRadius = g_DebugParticleFallbackRadius;
    if ((g_DebugParticleFlags & CRESSIM_DEBUG_PARTICLE_USE_RADII) != 0u)
    {
        particleRadius = CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex);
    }
    particleRadius = max(particleRadius, 1.0e-4);

    if (preparedCamera.active == 0u || cameraInput.active == 0u ||
        particleEnvIndex != g_DebugParticleEnvIndex)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.ViewPos = float3(0.0, 0.0, 1.0);
        Out.QuadCoord = float2(0.0, 0.0);
        Out.Radius = particleRadius;
#if MANUAL_LAYER_EXPORT
        Out.Layer = g_DebugParticleTargetLayer;
#endif
        return;
    }

    const float4 worldPos = float4(particlePositionInvMass.xyz, 1.0);
    const float3 viewCenter = mul(worldPos, preparedCamera.viewMatrix).xyz;
    const float2 quadCoord = quadCornerForVertex(triangleVertexIndex);
    const float3 viewPos = viewCenter + float3(quadCoord * particleRadius, 0.0);

    const float aspect =
        computeEffectiveViewportAspect(cameraInput.viewportAndOutputSize);
    const float4x4 projectionMatrix =
        buildProjectionMatrix(cameraInput.projectionParams.x, aspect,
                              cameraInput.projectionParams.y, cameraInput.projectionParams.z);

    Out.Position = mul(float4(viewPos, 1.0), projectionMatrix);
    Out.ViewPos = viewPos;
    Out.QuadCoord = quadCoord;
    Out.Radius = particleRadius;
#if MANUAL_LAYER_EXPORT
    Out.Layer = g_DebugParticleTargetLayer;
#endif
}
