#include "graphics/math/diligent_math_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cressim::neo::graphics::math
{

namespace
{

constexpr float kEpsilon = 1.0e-12f;

float clampPositive(float value, float fallback) noexcept
{
    return value > 0.0f ? value : fallback;
}

Diligent::float3 safeNormalize(const Diligent::float3& value) noexcept
{
    const float lengthSq = Diligent::dot(value, value);
    if (lengthSq <= kEpsilon)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float invLength = 1.0f / std::sqrt(lengthSq);
    return value * invLength;
}

Diligent::QuaternionF toQuaternion(const common::Quatf& value) noexcept
{
    const Diligent::QuaternionF quat{value.x, value.y, value.z, value.w};
    const float lengthSq = Diligent::dot(quat.q, quat.q);
    if (lengthSq <= kEpsilon)
    {
        return {};
    }
    return Diligent::normalize(quat);
}

} // namespace

Diligent::float3 toFloat3(const common::Vec3f& value) noexcept
{
    return {value.x, value.y, value.z};
}

Diligent::float4x4 transformMatrix(const common::Transform& transform) noexcept
{
    const Diligent::QuaternionF rotation = toQuaternion(transform.rotation);
    const Diligent::float3 right = rotation.RotateVector(Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 up = rotation.RotateVector(Diligent::float3{0.0f, 1.0f, 0.0f});
    const Diligent::float3 forward = rotation.RotateVector(Diligent::float3{0.0f, 0.0f, 1.0f});

    Diligent::float4x4 out = Diligent::float4x4::Identity();
    out.m00 = right.x * transform.scale.x;
    out.m01 = up.x * transform.scale.y;
    out.m02 = forward.x * transform.scale.z;
    out.m10 = right.y * transform.scale.x;
    out.m11 = up.y * transform.scale.y;
    out.m12 = forward.y * transform.scale.z;
    out.m20 = right.z * transform.scale.x;
    out.m21 = up.z * transform.scale.y;
    out.m22 = forward.z * transform.scale.z;
    out.m30 = transform.position.x;
    out.m31 = transform.position.y;
    out.m32 = transform.position.z;
    return out;
}

Diligent::float4x4 viewMatrixFromTransform(const common::Transform& cameraTransform) noexcept
{
    const Diligent::QuaternionF rotation = toQuaternion(cameraTransform.rotation);
    const Diligent::float3 eye = toFloat3(cameraTransform.position);
    const Diligent::float3 forward = safeNormalize(rotation.RotateVector(Diligent::float3{0.0f, 0.0f, -1.0f}));
    const Diligent::float3 up = safeNormalize(rotation.RotateVector(Diligent::float3{0.0f, 1.0f, 0.0f}));
    const Diligent::float3 at = eye + forward;

    const Diligent::float3 zAxis = safeNormalize(eye - at);
    const Diligent::float3 xAxis = safeNormalize(Diligent::cross(up, zAxis));
    const Diligent::float3 yAxis = Diligent::cross(zAxis, xAxis);

    Diligent::float4x4 out = Diligent::float4x4::Identity();
    out.m00 = xAxis.x;
    out.m01 = yAxis.x;
    out.m02 = zAxis.x;
    out.m10 = xAxis.y;
    out.m11 = yAxis.y;
    out.m12 = zAxis.y;
    out.m20 = xAxis.z;
    out.m21 = yAxis.z;
    out.m22 = zAxis.z;
    out.m30 = -Diligent::dot(xAxis, eye);
    out.m31 = -Diligent::dot(yAxis, eye);
    out.m32 = -Diligent::dot(zAxis, eye);
    return out;
}

Diligent::float4x4 perspectiveMatrix(float verticalFovDegrees, float aspectRatio, float nearClip, float farClip) noexcept
{
    const float fovRadians = verticalFovDegrees * 0.017453292519943295769f;
    const float tanHalfFov = std::tan(0.5f * fovRadians);
    const float yScale = 1.0f / clampPositive(tanHalfFov, 1.0f);
    const float xScale = yScale / clampPositive(aspectRatio, 1.0f);
    const float nearPlane = std::max(nearClip, 0.001f);
    const float farPlane = std::max(farClip, nearPlane + 0.001f);
    const float clipRange = nearPlane - farPlane;

    Diligent::float4x4 out{};
    out.m00 = xScale;
    out.m11 = yScale;
    out.m22 = farPlane / clipRange;
    out.m23 = -1.0f;
    out.m32 = (nearPlane * farPlane) / clipRange;
    return out;
}

void copyMatrixRowMajor(float dst[16], const Diligent::float4x4& matrix) noexcept
{
    std::memcpy(dst, matrix.Data(), 16 * sizeof(float));
}

} // namespace cressim::neo::graphics::math
