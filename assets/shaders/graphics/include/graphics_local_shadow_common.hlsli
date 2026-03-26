#ifndef CRESSIM_NEO_GRAPHICS_LOCAL_SHADOW_COMMON_HLSLI
#define CRESSIM_NEO_GRAPHICS_LOCAL_SHADOW_COMMON_HLSLI

#include "graphics/include/graphics_scene_buffers.hlsli"

static const float CRESSIM_LOCAL_SHADOW_PI = 3.14159265359f;
static const uint CRESSIM_LOCAL_SHADOW_POINT_FACE_COUNT = 6u;
static const uint CRESSIM_LOCAL_SHADOW_VIEWS_PER_ENV =
    CRESSIM_SHADOWED_LOCAL_LIGHT_CAP + CRESSIM_SHADOWED_POINT_LIGHT_CAP;
static const float CRESSIM_LOCAL_SHADOW_POINT_FACE_FOV_RADIANS =
    (90.0f + 5.0f) * (CRESSIM_LOCAL_SHADOW_PI / 180.0f);
static const float CRESSIM_LOCAL_SHADOW_DIRECTIONAL_DEPTH_PADDING = 4.0f;
static const uint CRESSIM_LOCAL_SHADOW_ENV_BOUNDS_WORDS = 8u;

float3 localShadowSafeNormalize(float3 v, float3 fallbackValue)
{
    const float lenSq = dot(v, v);
    return lenSq > 1e-6 ? v * rsqrt(lenSq) : fallbackValue;
}

float4x4 localShadowBuildLookAtMatrix(float3 eye, float3 at, float3 upCandidate)
{
    const float3 zAxis = localShadowSafeNormalize(at - eye, float3(0.0, 0.0, 1.0));
    const float3 xAxis = localShadowSafeNormalize(cross(upCandidate, zAxis), float3(1.0, 0.0, 0.0));
    const float3 yAxis = localShadowSafeNormalize(cross(zAxis, xAxis), float3(0.0, 1.0, 0.0));

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

float4x4 localShadowBuildPerspectiveMatrix(float verticalFovRadians, float aspect, float nearPlane,
                                           float farPlane)
{
    const float yScale = 1.0 / tan(0.5 * max(verticalFovRadians, 0.017453292519943295f));
    const float xScale = yScale / max(aspect, 1e-5);
    const float zScale = farPlane / max(farPlane - nearPlane, 1e-5);
    const float zTranslate = -nearPlane * farPlane / max(farPlane - nearPlane, 1e-5);

    return float4x4(
        xScale, 0.0,    0.0,       0.0,
        0.0,    yScale, 0.0,       0.0,
        0.0,    0.0,    zScale,    1.0,
        0.0,    0.0,    zTranslate, 0.0);
}

float4x4 localShadowBuildOrthoOffCenterMatrix(float left, float right, float bottom, float top,
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

uint localShadowTotalLightCount(uint envCount, uint maxLightsPerEnv)
{
    return envCount * maxLightsPerEnv;
}

uint localShadowTotalObjectCount(uint envCount, uint maxObjectsPerEnv)
{
    return envCount * maxObjectsPerEnv;
}

uint localShadowTotalViewCount(uint envCount)
{
    return envCount * CRESSIM_LOCAL_SHADOW_VIEWS_PER_ENV;
}

uint localShadowTotal2DSubviewCount(uint envCount)
{
    return envCount * CRESSIM_SHADOWED_LOCAL_LIGHT_CAP;
}

uint localShadowTotalPointFaceCount(uint envCount)
{
    return envCount * CRESSIM_SHADOWED_POINT_LIGHT_CAP * CRESSIM_LOCAL_SHADOW_POINT_FACE_COUNT;
}

uint localShadow2DViewIndex(uint envIndex, uint localSlot)
{
    return envIndex * CRESSIM_LOCAL_SHADOW_VIEWS_PER_ENV + localSlot;
}

uint localShadowPointViewIndex(uint envIndex, uint pointSlot)
{
    return envIndex * CRESSIM_LOCAL_SHADOW_VIEWS_PER_ENV +
           CRESSIM_SHADOWED_LOCAL_LIGHT_CAP + pointSlot;
}

uint localShadow2DLayer(uint envIndex, uint localSlot)
{
    return envIndex * CRESSIM_SHADOWED_LOCAL_LIGHT_CAP + localSlot;
}

uint localShadowPointFirstLayer(uint envIndex, uint pointSlot)
{
    return (envIndex * CRESSIM_SHADOWED_POINT_LIGHT_CAP + pointSlot) *
           CRESSIM_LOCAL_SHADOW_POINT_FACE_COUNT;
}

uint localShadow2DSubviewIndexToEnv(uint subviewIndex)
{
    return subviewIndex / CRESSIM_SHADOWED_LOCAL_LIGHT_CAP;
}

uint localShadow2DSubviewIndexToSlot(uint subviewIndex)
{
    return subviewIndex % CRESSIM_SHADOWED_LOCAL_LIGHT_CAP;
}

uint localShadowPointFaceIndexToEnv(uint faceIndex)
{
    return faceIndex / (CRESSIM_SHADOWED_POINT_LIGHT_CAP * CRESSIM_LOCAL_SHADOW_POINT_FACE_COUNT);
}

uint localShadowPointFaceIndexToPointSlot(uint faceIndex)
{
    return (faceIndex / CRESSIM_LOCAL_SHADOW_POINT_FACE_COUNT) % CRESSIM_SHADOWED_POINT_LIGHT_CAP;
}

uint localShadowPointFaceIndexToLocalFace(uint faceIndex)
{
    return faceIndex % CRESSIM_LOCAL_SHADOW_POINT_FACE_COUNT;
}

uint localShadowFloatToOrderedUint(float value)
{
    const uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

float localShadowOrderedUintToFloat(uint value)
{
    const uint bits = (value & 0x80000000u) != 0u ? (value & 0x7fffffffu) : ~value;
    return asfloat(bits);
}

void localShadowBuildRenderableCorners(uint objectIndex, out bool valid, out float3 corners[8])
{
    bool poseValid = false;
    float3 position = float3(0.0, 0.0, 0.0);
    float4 orientation = float4(0.0, 0.0, 0.0, 1.0);
    float3 scale = float3(1.0, 1.0, 1.0);
    loadRenderablePose(objectIndex, poseValid, position, orientation, scale);
    valid = poseValid;

    RenderableMetadata metadata = g_RenderableMetadata[objectIndex];
    corners[0] = metadata.localBoundsMin.xyz;
    corners[1] = float3(metadata.localBoundsMax.x, metadata.localBoundsMin.y, metadata.localBoundsMin.z);
    corners[2] = float3(metadata.localBoundsMax.x, metadata.localBoundsMax.y, metadata.localBoundsMin.z);
    corners[3] = float3(metadata.localBoundsMin.x, metadata.localBoundsMax.y, metadata.localBoundsMin.z);
    corners[4] = float3(metadata.localBoundsMin.x, metadata.localBoundsMin.y, metadata.localBoundsMax.z);
    corners[5] = float3(metadata.localBoundsMax.x, metadata.localBoundsMin.y, metadata.localBoundsMax.z);
    corners[6] = metadata.localBoundsMax.xyz;
    corners[7] = float3(metadata.localBoundsMin.x, metadata.localBoundsMax.y, metadata.localBoundsMax.z);

    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        corners[i] = quaternionRotateVector(orientation, corners[i] * scale) + position;
    }
}

bool localShadowMatrixIntersectsCorners(float3 corners[8], float4x4 viewProjectionMatrix)
{
    bool allLeft = true;
    bool allRight = true;
    bool allBottom = true;
    bool allTop = true;
    bool allNear = true;
    bool allFar = true;

    [unroll]
    for (int i = 0; i < 8; ++i)
    {
        const float4 clip = mul(float4(corners[i], 1.0), viewProjectionMatrix);
        allLeft = allLeft && (clip.x < -clip.w);
        allRight = allRight && (clip.x > clip.w);
        allBottom = allBottom && (clip.y < -clip.w);
        allTop = allTop && (clip.y > clip.w);
        allNear = allNear && (clip.z < 0.0);
        allFar = allFar && (clip.z > clip.w);
    }

    return !(allLeft || allRight || allBottom || allTop || allNear || allFar);
}

void localShadowDecodeEnvBounds(StructuredBuffer<uint> envBounds, uint envIndex, out bool valid,
                                out float3 center, out float radius)
{
    const uint baseIndex = envIndex * CRESSIM_LOCAL_SHADOW_ENV_BOUNDS_WORDS;
    valid = envBounds[baseIndex + 6u] != 0u;
    if (!valid)
    {
        center = float3(0.0, 0.0, 0.0);
        radius = 10.0;
        return;
    }

    const float3 minPoint = float3(
        localShadowOrderedUintToFloat(envBounds[baseIndex + 0u]),
        localShadowOrderedUintToFloat(envBounds[baseIndex + 1u]),
        localShadowOrderedUintToFloat(envBounds[baseIndex + 2u]));
    const float3 maxPoint = float3(
        localShadowOrderedUintToFloat(envBounds[baseIndex + 3u]),
        localShadowOrderedUintToFloat(envBounds[baseIndex + 4u]),
        localShadowOrderedUintToFloat(envBounds[baseIndex + 5u]));
    center = (minPoint + maxPoint) * 0.5;
    const float3 extents = maxPoint - minPoint;
    radius = max(5.0, length(extents) * 0.5 + 2.0);
}

#endif // CRESSIM_NEO_GRAPHICS_LOCAL_SHADOW_COMMON_HLSLI
