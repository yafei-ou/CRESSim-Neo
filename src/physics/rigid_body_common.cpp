#include "physics/rigid_body_common.h"

#include "common/math_utils_runtime.h"

#include <algorithm>
#include <cmath>

namespace cressim::neo::physics
{

namespace
{

Diligent::float3 absFloat3(const Diligent::float3 &value)
{
    return Diligent::float3{std::abs(value.x), std::abs(value.y), std::abs(value.z)};
}

Diligent::QuaternionF multiplyQuaternions(const Diligent::QuaternionF &a,
                                          const Diligent::QuaternionF &b)
{
    return Diligent::QuaternionF{a.q.w * b.q.x + a.q.x * b.q.w + a.q.y * b.q.z - a.q.z * b.q.y,
                                 a.q.w * b.q.y - a.q.x * b.q.z + a.q.y * b.q.w + a.q.z * b.q.x,
                                 a.q.w * b.q.z + a.q.x * b.q.y - a.q.y * b.q.x + a.q.z * b.q.w,
                                 a.q.w * b.q.w - a.q.x * b.q.x - a.q.y * b.q.y - a.q.z * b.q.z};
}

Diligent::QuaternionF conjugateQuaternion(const Diligent::QuaternionF &q)
{
    return Diligent::QuaternionF{-q.q.x, -q.q.y, -q.q.z, q.q.w};
}

} // namespace

std::uint32_t estimateRigidCandidatePairCapacity(std::uint32_t bodyCount) noexcept
{
    if (bodyCount < 2u)
    {
        return 0u;
    }

    constexpr std::uint32_t kPairsPerBodyEstimate = 32u;
    constexpr std::uint32_t kMinCapacity          = 64u;
    return std::max<std::uint32_t>(bodyCount * kPairsPerBodyEstimate, kMinCapacity);
}

EffectiveColliderDimensions computeEffectiveColliderDimensions(
    ColliderShapeType shape, const Diligent::float4 &colliderParams,
    const Diligent::float3 &scale) noexcept
{
    EffectiveColliderDimensions result{};
    const Diligent::float3 absScale = absFloat3(scale);

    switch (shape)
    {
    case ColliderShapeType::Sphere:
        result.sphereRadius = colliderParams.x * std::max({absScale.x, absScale.y, absScale.z});
        break;
    case ColliderShapeType::Box:
        result.boxHalfExtents =
            Diligent::float3{colliderParams.x * absScale.x, colliderParams.y * absScale.y,
                             colliderParams.z * absScale.z};
        break;
    case ColliderShapeType::Capsule:
        result.capsuleRadius     = colliderParams.x * std::max(absScale.x, absScale.z);
        result.capsuleHalfHeight = colliderParams.y * absScale.y;
        break;
    }

    return result;
}

Diligent::QuaternionF rotationVectorToQuaternion(const Diligent::float3 &rotationVector) noexcept
{
    const float angleSq = Diligent::dot(rotationVector, rotationVector);
    if (angleSq <= common::runtime_math::kEpsilon)
    {
        return common::runtime_math::normalizeQuaternion(Diligent::QuaternionF{
            rotationVector.x * 0.5f, rotationVector.y * 0.5f, rotationVector.z * 0.5f, 1.0f});
    }

    const float angle     = std::sqrt(angleSq);
    const float halfAngle = angle * 0.5f;
    const float scale     = std::sin(halfAngle) / angle;
    return common::runtime_math::normalizeQuaternion(
        Diligent::QuaternionF{rotationVector.x * scale, rotationVector.y * scale,
                              rotationVector.z * scale, std::cos(halfAngle)});
}

Diligent::QuaternionF integrateOrientation(const Diligent::QuaternionF &orientation,
                                           const Diligent::float3 &angularVelocity,
                                           float dt) noexcept
{
    if (dt <= common::runtime_math::kEpsilon)
    {
        return common::runtime_math::normalizeQuaternion(orientation);
    }

    const Diligent::QuaternionF delta = rotationVectorToQuaternion(angularVelocity * dt);
    return common::runtime_math::normalizeQuaternion(multiplyQuaternions(delta, orientation));
}

Diligent::float3 angularVelocityFromOrientationDelta(const Diligent::QuaternionF &previous,
                                                     const Diligent::QuaternionF &current,
                                                     float dt) noexcept
{
    if (dt <= common::runtime_math::kEpsilon)
    {
        return Diligent::float3{0.0f, 0.0f, 0.0f};
    }

    Diligent::QuaternionF delta = multiplyQuaternions(
        common::runtime_math::normalizeQuaternion(current),
        conjugateQuaternion(common::runtime_math::normalizeQuaternion(previous)));
    if (delta.q.w < 0.0f)
    {
        delta.q = -delta.q;
    }

    const Diligent::float3 imag{delta.q.x, delta.q.y, delta.q.z};
    const float imagLengthSq = Diligent::dot(imag, imag);
    if (imagLengthSq <= common::runtime_math::kEpsilon)
    {
        return imag * (2.0f / dt);
    }

    const float imagLength = std::sqrt(imagLengthSq);
    const float angle      = 2.0f * std::atan2(imagLength, delta.q.w);
    return imag * (angle / (imagLength * dt));
}

Diligent::float3 multiplyWorldInverseInertia(const Diligent::float3 &inverseInertiaLocal,
                                             const Diligent::QuaternionF &orientation,
                                             const Diligent::float3 &vector) noexcept
{
    const Diligent::float3 axisX = orientation.RotateVector(Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 axisY = orientation.RotateVector(Diligent::float3{0.0f, 1.0f, 0.0f});
    const Diligent::float3 axisZ = orientation.RotateVector(Diligent::float3{0.0f, 0.0f, 1.0f});

    return axisX * (inverseInertiaLocal.x * Diligent::dot(axisX, vector)) +
           axisY * (inverseInertiaLocal.y * Diligent::dot(axisY, vector)) +
           axisZ * (inverseInertiaLocal.z * Diligent::dot(axisZ, vector));
}

} // namespace cressim::neo::physics
