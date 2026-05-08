#include "include/graphics/graphics_scene_buffers.hlsli"

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

cbuffer GraphicsCameraPrepareConstants
{
    uint g_CameraCount;
    uint g_MaxObjectsPerEnv;
    uint g_MaxLightsPerEnv;
    uint g_ShadowMapResolution;
};

CRESSIM_STRUCTURED_BUFFER(CameraInput, g_CameraInputs);
CRESSIM_RW_STRUCTURED_BUFFER(PreparedCamera, g_PreparedCamerasRW);

static const float PI = 3.14159265359f;
static const float kCascadeSplitLambda = 0.50f;
static const float kCascadeStabilization = 16.0f;
static const float kCascadeDepthPadding = 16.0f;
static const float kCascadeCasterExtrusion = 96.0f;
static const uint kShadowCascadeCount = 4u;

float3 safeNormalize(float3 v, float3 fallbackValue)
{
    const float lenSq = dot(v, v);
    return lenSq > 1e-6 ? v * rsqrt(lenSq) : fallbackValue;
}

float degreesToRadians(float degrees)
{
    return degrees * 0.017453292519943295f;
}

float3x3 identity3x3()
{
    return float3x3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
}

float4x4 buildViewMatrix(float3 position, float4 orientation)
{
    const float3 forward = safeNormalize(quaternionRotateVector(orientation, float3(0.0, 0.0, 1.0)),
                                         float3(0.0, 0.0, 1.0));
    float3 up = safeNormalize(quaternionRotateVector(orientation, float3(0.0, 1.0, 0.0)),
                              float3(0.0, 1.0, 0.0));
    const float3 right = safeNormalize(cross(up, forward), float3(1.0, 0.0, 0.0));
    up = safeNormalize(cross(forward, right), float3(0.0, 1.0, 0.0));

    const float4x4 viewRotation = float4x4(
        right.x, up.x, forward.x, 0.0,
        right.y, up.y, forward.y, 0.0,
        right.z, up.z, forward.z, 0.0,
        0.0,     0.0,  0.0,       1.0);
    const float4x4 viewTranslation = float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -position.x, -position.y, -position.z, 1.0);
    return mul(viewTranslation, viewRotation);
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

void computeCascadeSplits(float nearPlane, float farPlane, out float4 outSplits)
{
    outSplits = float4(farPlane, farPlane, farPlane, farPlane);
    [unroll]
    for (uint i = 0u; i < kShadowCascadeCount; ++i)
    {
        const float p = (float)(i + 1u) / (float)kShadowCascadeCount;
        const float logarithmic = nearPlane * pow(farPlane / max(nearPlane, 1e-5), p);
        const float linearSplit = nearPlane + (farPlane - nearPlane) * p;
        outSplits[i] = lerp(linearSplit, logarithmic, kCascadeSplitLambda);
    }
}

float4x4 buildLookAtMatrix(float3 eye, float3 at, float3 upCandidate)
{
    const float3 zAxis = safeNormalize(at - eye, float3(0.0, 0.0, 1.0));
    const float3 xAxis = safeNormalize(cross(upCandidate, zAxis), float3(1.0, 0.0, 0.0));
    const float3 yAxis = safeNormalize(cross(zAxis, xAxis), float3(0.0, 1.0, 0.0));

    const float4x4 viewRotation = float4x4(
        xAxis.x, yAxis.x, zAxis.x, 0.0,
        xAxis.y, yAxis.y, zAxis.y, 0.0,
        xAxis.z, yAxis.z, zAxis.z, 0.0,
        0.0,     0.0,     0.0,     1.0);
    const float4x4 viewTranslation = float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -eye.x, -eye.y, -eye.z, 1.0);
    return mul(viewTranslation, viewRotation);
}

float4x4 buildOrthoOffCenterMatrix(float left, float right, float bottom, float top,
                                   float nearPlane, float farPlane)
{
    const float invWidth = 1.0 / max(right - left, 1e-5);
    const float invHeight = 1.0 / max(top - bottom, 1e-5);
    const float invDepth = 1.0 / max(farPlane - nearPlane, 1e-5);

    return float4x4(
        2.0 * invWidth, 0.0, 0.0, 0.0,
        0.0, 2.0 * invHeight, 0.0, 0.0,
        0.0, 0.0, invDepth, 0.0,
        -(right + left) * invWidth, -(top + bottom) * invHeight,
        -nearPlane * invDepth, 1.0);
}

void buildFrustumCornersForRange(float3 cameraPosition, float3 cameraRight, float3 cameraUp,
                                 float3 cameraForward, float aspect, float fovRadians,
                                 float splitNear, float splitFar, out float3 corners[8])
{
    const float tanHalfFov = tan(fovRadians * 0.5);
    const float nearHalfHeight = tanHalfFov * splitNear;
    const float nearHalfWidth = nearHalfHeight * aspect;
    const float farHalfHeight = tanHalfFov * splitFar;
    const float farHalfWidth = farHalfHeight * aspect;

    const float3 nearCenter = cameraPosition + cameraForward * splitNear;
    const float3 farCenter = cameraPosition + cameraForward * splitFar;

    corners[0] = nearCenter - cameraRight * nearHalfWidth - cameraUp * nearHalfHeight;
    corners[1] = nearCenter + cameraRight * nearHalfWidth - cameraUp * nearHalfHeight;
    corners[2] = nearCenter + cameraRight * nearHalfWidth + cameraUp * nearHalfHeight;
    corners[3] = nearCenter - cameraRight * nearHalfWidth + cameraUp * nearHalfHeight;
    corners[4] = farCenter - cameraRight * farHalfWidth - cameraUp * farHalfHeight;
    corners[5] = farCenter + cameraRight * farHalfWidth - cameraUp * farHalfHeight;
    corners[6] = farCenter + cameraRight * farHalfWidth + cameraUp * farHalfHeight;
    corners[7] = farCenter - cameraRight * farHalfWidth + cameraUp * farHalfHeight;
}

float4x4 buildDirectionalLightCascadeViewProjection(float3 lightDirection, float3 corners[8])
{
    float3 cascadeCenter = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        cascadeCenter += corners[i];
    }
    cascadeCenter /= 8.0;

    float radius = 1.0;
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        radius = max(radius, length(corners[i] - cascadeCenter));
    }
    radius = ceil(radius * kCascadeStabilization) / kCascadeStabilization;

    const float3 upCandidate =
        abs(dot(lightDirection, float3(0.0, 1.0, 0.0))) > 0.98
            ? float3(0.0, 0.0, 1.0)
            : float3(0.0, 1.0, 0.0);
    const float3 lightPosition =
        cascadeCenter - lightDirection * (radius * 2.0 + kCascadeDepthPadding);
    const float4x4 lightView = buildLookAtMatrix(lightPosition, cascadeCenter, upCandidate);

    const float4 lightSpaceCenter = mul(float4(cascadeCenter, 1.0), lightView);
    const float texelWorldSize = (radius * 2.0) / max((float)g_ShadowMapResolution, 1.0);
    float snappedCenterX = lightSpaceCenter.x;
    float snappedCenterY = lightSpaceCenter.y;
    if (texelWorldSize > 0.0)
    {
        snappedCenterX = floor(snappedCenterX / texelWorldSize + 0.5) * texelWorldSize;
        snappedCenterY = floor(snappedCenterY / texelWorldSize + 0.5) * texelWorldSize;
    }

    float minZ = 3.402823466e+38f;
    float maxZ = -3.402823466e+38f;
    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        const float4 lightSpaceCorner = mul(float4(corners[i], 1.0), lightView);
        minZ = min(minZ, lightSpaceCorner.z);
        maxZ = max(maxZ, lightSpaceCorner.z);
    }

    const float left = snappedCenterX - radius;
    const float right = snappedCenterX + radius;
    const float bottom = snappedCenterY - radius;
    const float top = snappedCenterY + radius;
    // Directional cascades need extra depth on the light-facing side so casters that sit
    // outside the camera slice can still project onto receivers inside it.
    const float nearPlane = max(0.1, minZ - (kCascadeDepthPadding + kCascadeCasterExtrusion));
    const float farPlane = max(nearPlane + 1.0, maxZ + kCascadeDepthPadding);

    return mul(lightView, buildOrthoOffCenterMatrix(left, right, bottom, top, nearPlane, farPlane));
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint currentCameraIndex = dispatchThreadId.x;
    if (currentCameraIndex >= g_CameraCount)
    {
        return;
    }

    const CameraInput camera = CRESSIM_SB_LOAD(g_CameraInputs, currentCameraIndex);
    PreparedCamera prepared = (PreparedCamera)0;
    prepared.viewMatrix = float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
    prepared.viewProjectionMatrix = prepared.viewMatrix;
    [unroll]
    for (uint cascadeIndex = 0u; cascadeIndex < kShadowCascadeCount; ++cascadeIndex)
    {
        prepared.lightViewProjectionMatrices[cascadeIndex] = prepared.viewMatrix;
    }
    prepared.envIndex = camera.envIndex;
    prepared.active = camera.active;
    prepared.objectRangeStart = camera.envIndex * g_MaxObjectsPerEnv;
    prepared.objectRangeCount = g_MaxObjectsPerEnv;
    prepared.visibilityDataOffset = currentCameraIndex * g_MaxObjectsPerEnv;
    prepared.cascadeSplits = float4(0.0, 0.0, 0.0, 0.0);
    prepared.mainShadowTexelSize = float2(0.0, 0.0);
    prepared.mainShadowCascadeCount = 0.0;
    prepared.mainShadowFadeDistance = 0.0;

    if (camera.active == 0u)
    {
        CRESSIM_SB_STORE(g_PreparedCamerasRW, currentCameraIndex, prepared);
        return;
    }

    const float3 position = camera.position.xyz;
    const float4 orientation = normalize(camera.orientation);
    const float aspect = computeEffectiveViewportAspect(camera.viewportAndOutputSize);
    const float nearClip = max(camera.projectionParams.y, 0.001);
    const float farClip = max(camera.projectionParams.z, nearClip + 0.001);
    const float fovRadians = max(degreesToRadians(camera.projectionParams.x), degreesToRadians(1.0));
    const float3 cameraForward =
        safeNormalize(quaternionRotateVector(orientation, float3(0.0, 0.0, 1.0)),
                      float3(0.0, 0.0, 1.0));
    const float3 worldUp =
        safeNormalize(quaternionRotateVector(orientation, float3(0.0, 1.0, 0.0)),
                      float3(0.0, 1.0, 0.0));
    const float3 cameraRight = safeNormalize(cross(worldUp, cameraForward), float3(1.0, 0.0, 0.0));
    const float3 cameraUp = safeNormalize(cross(cameraForward, cameraRight), float3(0.0, 1.0, 0.0));

    prepared.viewMatrix = buildViewMatrix(position, orientation);
    const float4x4 projectionMatrix =
        buildProjectionMatrix(camera.projectionParams.x, aspect, nearClip, farClip);
    prepared.viewProjectionMatrix = mul(prepared.viewMatrix, projectionMatrix);
    prepared.cameraPosition = float4(position, 1.0);

    LightInput light = (LightInput)0;
    bool hasDirectionalLight = false;
    if (g_MaxLightsPerEnv > 0u)
    {
        const uint lightIndex = camera.envIndex * g_MaxLightsPerEnv;
        light = CRESSIM_SB_LOAD(g_LightInputs, lightIndex);
        hasDirectionalLight =
            light.active != 0u &&
            dot(light.directionIntensity.xyz, light.directionIntensity.xyz) > 1e-6 &&
            light.directionIntensity.w > 0.0;
    }
    if (hasDirectionalLight && light.castsShadows != 0u)
    {
        const float3 lightDirection = safeNormalize(light.directionIntensity.xyz, float3(0.0, -1.0, 0.0));
        const float shadowDistance = min(farClip, max(light.shadowDistance, nearClip + 0.001));
        computeCascadeSplits(nearClip, shadowDistance, prepared.cascadeSplits);
        prepared.mainShadowTexelSize = float2(1.0 / max((float)g_ShadowMapResolution, 1.0),
                                              1.0 / max((float)g_ShadowMapResolution, 1.0));
        prepared.mainShadowCascadeCount = (float)kShadowCascadeCount;
        prepared.mainShadowFadeDistance = max(light.shadowFadeDistance, 0.001);

        float splitNear = nearClip;
        [unroll]
        for (uint cascadeIndex = 0u; cascadeIndex < kShadowCascadeCount; ++cascadeIndex)
        {
            float3 cascadeCorners[8];
            const float splitFar = prepared.cascadeSplits[cascadeIndex];
            buildFrustumCornersForRange(position, cameraRight, cameraUp, cameraForward, aspect,
                                        fovRadians, splitNear, splitFar, cascadeCorners);
            prepared.lightViewProjectionMatrices[cascadeIndex] =
                buildDirectionalLightCascadeViewProjection(lightDirection, cascadeCorners);
            splitNear = splitFar;
        }
    }

    CRESSIM_SB_STORE(g_PreparedCamerasRW, currentCameraIndex, prepared);
}
