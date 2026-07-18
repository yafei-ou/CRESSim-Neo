#include "graphics_scene_buffers.hlsli"
#include "graphics_camera_input.hlsli"
#include "physics_rigid_types.hlsli"

cbuffer GraphicsDebugRoutedCables
{
    uint4 g_DebugRoutedCableParams;
    float4 g_DebugRoutedCableMisc;
};

#define g_DebugRoutedCableCameraIndex g_DebugRoutedCableParams.x
#define g_DebugRoutedCableTargetLayer g_DebugRoutedCableParams.y
#define g_DebugRoutedCableEnvIndex g_DebugRoutedCableParams.z
#define g_DebugRoutedCableFlags g_DebugRoutedCableParams.w
#define g_DebugRoutedCableRadius g_DebugRoutedCableMisc.x
#define g_DebugRoutedCableOpacity g_DebugRoutedCableMisc.y

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(GpuRoutedCableRoutePoint, g_RoutedCableRoutePoints);
CRESSIM_STRUCTURED_BUFFER(GpuRoutedCableDebugSegment, g_RoutedCableDebugSegments);

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

float4 cableColor(uint cableIndex, float opacity)
{
    const uint paletteIndex = cableIndex % 6u;
    if (paletteIndex == 0u) return float4(0.95, 0.28, 0.18, opacity);
    if (paletteIndex == 1u) return float4(0.98, 0.78, 0.16, opacity);
    if (paletteIndex == 2u) return float4(0.18, 0.86, 0.72, opacity);
    if (paletteIndex == 3u) return float4(0.35, 0.58, 0.96, opacity);
    if (paletteIndex == 4u) return float4(0.92, 0.36, 0.80, opacity);
    return float4(0.92, 0.92, 0.92, opacity);
}

void main(uint vertexId : SV_VertexID, out VSOutput Out)
{
    const uint segmentIndex = vertexId / 6u;
    const uint triangleVertexIndex = vertexId % 6u;
    const PreparedCamera preparedCamera =
        CRESSIM_SB_LOAD(g_PreparedCameras, g_DebugRoutedCableCameraIndex);
    const CameraInput cameraInput = CRESSIM_SB_LOAD(g_CameraInputs, g_DebugRoutedCableCameraIndex);
    const GpuRoutedCableDebugSegment segment =
        CRESSIM_SB_LOAD(g_RoutedCableDebugSegments, segmentIndex);

    if (preparedCamera.active == 0u || cameraInput.active == 0u ||
        segment.envIndex != g_DebugRoutedCableEnvIndex)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.Color = float4(0.0, 0.0, 0.0, 0.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = g_DebugRoutedCableTargetLayer;
#endif
        return;
    }

    const GpuRoutedCableRoutePoint routePointA =
        CRESSIM_SB_LOAD(g_RoutedCableRoutePoints, segment.routePointIndexA);
    const GpuRoutedCableRoutePoint routePointB =
        CRESSIM_SB_LOAD(g_RoutedCableRoutePoints, segment.routePointIndexB);
    const float4 posInvMassA =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, routePointA.rigidBodyIndex);
    const float4 posInvMassB =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, routePointB.rigidBodyIndex);
    const float4 orientationA =
        normalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, routePointA.rigidBodyIndex));
    const float4 orientationB =
        normalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, routePointB.rigidBodyIndex));
    const float3 worldPointA =
        posInvMassA.xyz + quaternionRotateVector(orientationA, routePointA.localGuideOffset.xyz);
    const float3 worldPointB =
        posInvMassB.xyz + quaternionRotateVector(orientationB, routePointB.localGuideOffset.xyz);

    const float3 viewPointA = mul(float4(worldPointA, 1.0), preparedCamera.viewMatrix).xyz;
    const float3 viewPointB = mul(float4(worldPointB, 1.0), preparedCamera.viewMatrix).xyz;
    const float3 segmentView = viewPointB - viewPointA;
    const float segmentLengthSq = dot(segmentView, segmentView);
    if (segmentLengthSq <= 1.0e-8)
    {
        Out.Position = float4(2.0, 2.0, 2.0, 1.0);
        Out.Color = float4(0.0, 0.0, 0.0, 0.0);
#if MANUAL_LAYER_EXPORT
        Out.Layer = g_DebugRoutedCableTargetLayer;
#endif
        return;
    }

    const float3 segmentDir = segmentView * rsqrt(segmentLengthSq);
    float3 side = cross(float3(0.0, 0.0, 1.0), segmentDir);
    if (dot(side, side) <= 1.0e-8)
    {
        side = float3(1.0, 0.0, 0.0);
    }
    side = normalize(side) * max(g_DebugRoutedCableRadius, 1.0e-4);

    const float2 quadCorner = quadCornerForVertex(triangleVertexIndex);
    const float3 baseViewPos = quadCorner.y < 0.0 ? viewPointA : viewPointB;
    const float3 viewPos = baseViewPos + side * quadCorner.x;

    const float aspect =
        computeEffectiveViewportAspect(cameraInput.viewportAndOutputSize);
    const float4x4 projectionMatrix =
        buildProjectionMatrix(cameraInput.projectionParams.x, aspect,
                              cameraInput.projectionParams.y, cameraInput.projectionParams.z);

    Out.Position = mul(float4(viewPos, 1.0), projectionMatrix);
    Out.Color = cableColor(segment.cableIndex, g_DebugRoutedCableOpacity);
#if MANUAL_LAYER_EXPORT
    Out.Layer = g_DebugRoutedCableTargetLayer;
#endif
}
