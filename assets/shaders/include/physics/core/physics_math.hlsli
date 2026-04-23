#ifndef CRESSIM_NEO_PHYSICS_CORE_MATH_HLSLI
#define CRESSIM_NEO_PHYSICS_CORE_MATH_HLSLI

#include "physics_base.hlsli"

float3 SafeNormalize(float3 value, float3 fallback)
{
    const float lengthSq = dot(value, value);
    if (lengthSq <= kEpsilon)
        return fallback;
    return value * rsqrt(lengthSq);
}

float4 QuaternionNormalize(float4 q)
{
    const float lengthSq = dot(q, q);
    if (lengthSq <= kEpsilon)
        return float4(0.0, 0.0, 0.0, 1.0);
    return q * rsqrt(lengthSq);
}

float4 QuaternionConjugate(float4 q) { return float4(-q.xyz, q.w); }

float4 QuaternionMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 QuaternionRotate(float4 q, float3 v)
{
    const float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 ComposeColliderWorldPosition(float3 bodyPosition, float4 bodyOrientation,
                                    float3 colliderLocalPosition)
{
    return bodyPosition + QuaternionRotate(bodyOrientation, colliderLocalPosition);
}

float4 ComposeColliderWorldOrientation(float4 bodyOrientation, float4 colliderLocalOrientation)
{
    return QuaternionNormalize(QuaternionMul(bodyOrientation, colliderLocalOrientation));
}

float3 QuaternionInverseRotate(float4 q, float3 v)
{
    return QuaternionRotate(QuaternionConjugate(q), v);
}

float4 QuaternionFromRotationVector(float3 rotationVector)
{
    const float angleSq = dot(rotationVector, rotationVector);
    if (angleSq <= kEpsilon)
    {
        return QuaternionNormalize(float4(rotationVector * 0.5, 1.0));
    }

    const float angle = sqrt(angleSq);
    const float halfAngle = 0.5 * angle;
    const float scale = sin(halfAngle) / angle;
    return QuaternionNormalize(float4(rotationVector * scale, cos(halfAngle)));
}

float4 IntegrateOrientation(float4 orientation, float3 angularVelocity, float dt)
{
    return QuaternionNormalize(QuaternionMul(QuaternionFromRotationVector(angularVelocity * dt),
                                             orientation));
}

float3 AngularVelocityFromQuaternionDelta(float4 previous, float4 current, float dt)
{
    if (dt <= kEpsilon)
        return float3(0.0, 0.0, 0.0);

    float4 delta =
        QuaternionMul(QuaternionNormalize(current),
                      QuaternionConjugate(QuaternionNormalize(previous)));
    if (delta.w < 0.0)
        delta = -delta;

    const float imagLenSq = dot(delta.xyz, delta.xyz);
    if (imagLenSq <= kEpsilon)
        return delta.xyz * (2.0 / dt);

    const float imagLen = sqrt(imagLenSq);
    const float angle = 2.0 * atan2(imagLen, delta.w);
    return delta.xyz * (angle / (imagLen * dt));
}

float3 Abs3(float3 v) { return abs(v); }

#endif // CRESSIM_NEO_PHYSICS_CORE_MATH_HLSLI
