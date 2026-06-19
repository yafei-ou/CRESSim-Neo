#include "graphics_scene_buffers.hlsli"
#include "graphics_camera_input.hlsli"
#include "physics_particle_types.hlsli"

cbuffer GraphicsDebugParticles
{
    float4 g_DebugParticleColor;
    float4 g_DebugParticleStaticColor;
    float4 g_DebugParticleEdgeColor;
    float4 g_DebugParticleEdgeHighStrainColor;
    float4 g_DebugParticleEdgeDamagedColor;
    float4 g_DebugParticleEdgeDisabledColor;
    uint4 g_DebugParticleParams;
    uint4 g_DebugShapeParams;
    float4 g_DebugParticleMisc;
};

#define g_DebugParticleCameraIndex g_DebugParticleParams.x
#define g_DebugParticleTargetLayer g_DebugParticleParams.y
#define g_DebugParticleEnvIndex g_DebugParticleParams.z
#define g_DebugParticleFlags g_DebugParticleParams.w
#define g_DebugEdgeHighStrainThreshold g_DebugParticleMisc.y
#define g_DebugEdgeDamageDisplayThreshold g_DebugParticleMisc.z
#define CRESSIM_DEBUG_EDGE_SHOW_CUT 4u
#define CRESSIM_DEBUG_EDGE_SHOW_STRAIN 8u
#define CRESSIM_DEBUG_EDGE_SHOW_DAMAGE 16u

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(GpuSoftEdge, g_SoftEdges);

struct VSOutput
{
    float4 Position : SV_Position;
    nointerpolation float4 Color : TEXCOORD0;
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

void main(uint vertexId : SV_VertexID, out VSOutput Out)
{
    const uint edgeIndex = vertexId / 2u;
    const uint endpointIndex = vertexId & 1u;
    const GpuSoftEdge edge = CRESSIM_SB_LOAD(g_SoftEdges, edgeIndex);
    const uint particleIndex = endpointIndex == 0u ? edge.particleA : edge.particleB;
    const bool disabled = (edge.flags & kSoftEdgeDisabledFlag) != 0u ||
                          (edge.flags & kSoftEdgeActiveFlag) == 0u;
    const bool showDisabled = (g_DebugParticleFlags & CRESSIM_DEBUG_EDGE_SHOW_CUT) != 0u;

    const PreparedCamera preparedCamera = CRESSIM_SB_LOAD(g_PreparedCameras, g_DebugParticleCameraIndex);
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_DebugParticleCameraIndex);
    const uint particleEnvIndex = CRESSIM_SB_LOAD(g_ParticleEnvironmentIndices, particleIndex);

    if ((disabled && !showDisabled) ||
        preparedCamera.active == 0u || cameraInput.active == 0u ||
        particleEnvIndex != g_DebugParticleEnvIndex)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.Color = float4(0.0, 0.0, 0.0, 0.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = g_DebugParticleTargetLayer;
#endif
        return;
    }

    const float4 particlePositionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float3 viewPos = mul(float4(particlePositionInvMass.xyz, 1.0), preparedCamera.viewMatrix).xyz;
    const float aspect = computeEffectiveViewportAspect(cameraInput.viewportAndOutputSize);
    const float4x4 projectionMatrix =
        buildProjectionMatrix(cameraInput.projectionParams.x, aspect,
                              cameraInput.projectionParams.y, cameraInput.projectionParams.z);

    Out.Position = mul(float4(viewPos, 1.0), projectionMatrix);
    Out.Color = g_DebugParticleEdgeColor;
    if (disabled)
    {
        Out.Color = g_DebugParticleEdgeDisabledColor;
    }
    else if ((g_DebugParticleFlags & CRESSIM_DEBUG_EDGE_SHOW_DAMAGE) != 0u &&
             edge.damage >= g_DebugEdgeDamageDisplayThreshold)
    {
        Out.Color = g_DebugParticleEdgeDamagedColor;
    }
    else if ((g_DebugParticleFlags & CRESSIM_DEBUG_EDGE_SHOW_STRAIN) != 0u &&
             abs(edge.strain) >= g_DebugEdgeHighStrainThreshold)
    {
        Out.Color = g_DebugParticleEdgeHighStrainColor;
    }
#if MANUAL_LAYER_EXPORT
    Out.Layer = g_DebugParticleTargetLayer;
#endif
}
