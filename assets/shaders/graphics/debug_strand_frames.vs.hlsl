#include "graphics/graphics_scene_buffers.hlsli"
#include "graphics/graphics_camera_input.hlsli"
#include "physics/particle/physics_particle_types.hlsli"

cbuffer GraphicsDebugStrandFrames
{
    uint4 g_DebugStrandFrameParams;
    float4 g_DebugStrandFrameMisc;
};

#define g_DebugStrandFrameCameraIndex g_DebugStrandFrameParams.x
#define g_DebugStrandFrameTargetLayer g_DebugStrandFrameParams.y
#define g_DebugStrandFrameEnvIndex g_DebugStrandFrameParams.z
#define g_DebugStrandFrameAxisLength g_DebugStrandFrameMisc.x
#define g_DebugStrandFrameThickness g_DebugStrandFrameMisc.y
#define g_DebugStrandFrameOpacity g_DebugStrandFrameMisc.z

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleEnvironmentIndices);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegment, g_StrandSegments);
CRESSIM_STRUCTURED_BUFFER(GpuStrandSegmentState, g_StrandSegmentStates);

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
    const float xScale = yScale / max(aspect, 1.0e-5);
    const float zScale = farClip / max(farClip - nearClip, 1.0e-5);
    const float zTranslate = -nearClip * farClip / max(farClip - nearClip, 1.0e-5);

    return float4x4(
        xScale, 0.0,    0.0,       0.0,
        0.0,    yScale, 0.0,       0.0,
        0.0,    0.0,    zScale,    1.0,
        0.0,    0.0,    zTranslate, 0.0);
}

float2 quadCornerForVertex(uint triangleVertexIndex)
{
    if (triangleVertexIndex == 0u) return float2(-1.0, -1.0);
    if (triangleVertexIndex == 1u) return float2(1.0, -1.0);
    if (triangleVertexIndex == 2u) return float2(-1.0, 1.0);
    if (triangleVertexIndex == 3u) return float2(-1.0, 1.0);
    if (triangleVertexIndex == 4u) return float2(1.0, -1.0);
    return float2(1.0, 1.0);
}

float3 axisColor(uint axisIndex)
{
    if (axisIndex == 0u) return float3(1.0, 0.22, 0.18);
    if (axisIndex == 1u) return float3(0.22, 1.0, 0.30);
    return float3(0.24, 0.55, 1.0);
}

float3 axisDirection(float4 orientation, uint axisIndex)
{
    if (axisIndex == 0u) return quaternionRotateVector(orientation, float3(1.0, 0.0, 0.0));
    if (axisIndex == 1u) return quaternionRotateVector(orientation, float3(0.0, 1.0, 0.0));
    return quaternionRotateVector(orientation, float3(0.0, 0.0, 1.0));
}

void main(uint vertexId : SV_VertexID, out VSOutput Out)
{
    const uint segmentIndex = vertexId / 18u;
    const uint axisVertex = vertexId % 18u;
    const uint axisIndex = axisVertex / 6u;
    const uint triangleVertexIndex = axisVertex % 6u;

    const PreparedCamera preparedCamera =
        CRESSIM_SB_LOAD(g_PreparedCameras, g_DebugStrandFrameCameraIndex);
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_DebugStrandFrameCameraIndex);
    const GpuStrandSegment segment = CRESSIM_SB_LOAD(g_StrandSegments, segmentIndex);
    const GpuStrandSegmentState segmentState = CRESSIM_SB_LOAD(g_StrandSegmentStates, segmentIndex);

    if (preparedCamera.active == 0u || cameraInput.active == 0u ||
        CRESSIM_SB_LOAD(g_ParticleEnvironmentIndices, segment.particleA) != g_DebugStrandFrameEnvIndex)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.Color = float4(0.0, 0.0, 0.0, 0.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = g_DebugStrandFrameTargetLayer;
#endif
        return;
    }

    const float3 worldA = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleA).xyz;
    const float3 worldB = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, segment.particleB).xyz;
    const float3 midpoint = 0.5 * (worldA + worldB);
    const float4 orientation = normalize(segmentState.orientation);
    const float3 axisDirWorld = axisDirection(orientation, axisIndex);
    const float3 worldPointA = midpoint - 0.5 * g_DebugStrandFrameAxisLength * axisDirWorld;
    const float3 worldPointB = midpoint + 0.5 * g_DebugStrandFrameAxisLength * axisDirWorld;

    const float3 viewPointA = mul(float4(worldPointA, 1.0), preparedCamera.viewMatrix).xyz;
    const float3 viewPointB = mul(float4(worldPointB, 1.0), preparedCamera.viewMatrix).xyz;
    const float3 segmentView = viewPointB - viewPointA;
    const float segmentLengthSq = dot(segmentView, segmentView);
    if (segmentLengthSq <= 1.0e-8)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.Color = float4(0.0, 0.0, 0.0, 0.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = g_DebugStrandFrameTargetLayer;
#endif
        return;
    }

    const float3 segmentDir = segmentView * rsqrt(segmentLengthSq);
    float3 side = cross(float3(0.0, 0.0, 1.0), segmentDir);
    if (dot(side, side) <= 1.0e-8)
    {
        side = float3(1.0, 0.0, 0.0);
    }
    side = normalize(side) * max(g_DebugStrandFrameThickness, 1.0e-4);

    const float2 quadCorner = quadCornerForVertex(triangleVertexIndex);
    const float3 baseViewPos = quadCorner.y < 0.0 ? viewPointA : viewPointB;
    const float3 viewPos = baseViewPos + side * quadCorner.x;

    const float aspect = computeEffectiveViewportAspect(cameraInput.viewportAndOutputSize);
    const float4x4 projectionMatrix =
        buildProjectionMatrix(cameraInput.projectionParams.x, aspect,
                              cameraInput.projectionParams.y, cameraInput.projectionParams.z);

    Out.Position = mul(float4(viewPos, 1.0), projectionMatrix);
    Out.Color = float4(axisColor(axisIndex), g_DebugStrandFrameOpacity);
#if MANUAL_LAYER_EXPORT
    Out.Layer = g_DebugStrandFrameTargetLayer;
#endif
}
