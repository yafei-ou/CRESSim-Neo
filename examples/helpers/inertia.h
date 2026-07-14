#ifndef CRESSIM_NEO_EXAMPLES_HELPERS_INERTIA_H
#define CRESSIM_NEO_EXAMPLES_HELPERS_INERTIA_H

#include "physics/physics_types.h"

namespace cressim::neo::examples::helpers
{

inline Diligent::float3 computeBoxInverseInertia(const Diligent::float3 &halfExtents,
                                                 float inverseMass)
{
    if (inverseMass <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float ix = mass * (halfExtents.y * halfExtents.y + halfExtents.z * halfExtents.z) / 3.0f;
    const float iy = mass * (halfExtents.x * halfExtents.x + halfExtents.z * halfExtents.z) / 3.0f;
    const float iz = mass * (halfExtents.x * halfExtents.x + halfExtents.y * halfExtents.y) / 3.0f;
    return {ix > 0.0f ? 1.0f / ix : 0.0f, iy > 0.0f ? 1.0f / iy : 0.0f,
            iz > 0.0f ? 1.0f / iz : 0.0f};
}

inline Diligent::float3 computeSphereInverseInertia(float radius, float inverseMass)
{
    if (inverseMass <= 0.0f || radius <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float inertia = 0.4f * mass * radius * radius;
    const float inverseInertia = inertia > 0.0f ? 1.0f / inertia : 0.0f;
    return {inverseInertia, inverseInertia, inverseInertia};
}

inline Diligent::float3 computeApproximateCapsuleInverseInertia(float radius,
                                                                float halfHeight,
                                                                float inverseMass)
{
    return computeBoxInverseInertia({radius, halfHeight + radius, radius}, inverseMass);
}

inline Diligent::float3 computeCylinderInverseInertia(float radius, float halfHeight,
                                                      float inverseMass)
{
    if (inverseMass <= 0.0f || radius <= 0.0f || halfHeight <= 0.0f)
    {
        return {0.0f, 0.0f, 0.0f};
    }

    const float mass = 1.0f / inverseMass;
    const float height = 2.0f * halfHeight;
    const float inertiaRadial = (mass * (3.0f * radius * radius + height * height)) / 12.0f;
    const float inertiaAxial = 0.5f * mass * radius * radius;
    return {inertiaRadial > 0.0f ? 1.0f / inertiaRadial : 0.0f,
            inertiaAxial > 0.0f ? 1.0f / inertiaAxial : 0.0f,
            inertiaRadial > 0.0f ? 1.0f / inertiaRadial : 0.0f};
}

inline Diligent::float3 computeInverseInertiaForShape(physics::ColliderShapeType shapeType,
                                                      const Diligent::float4 &shapeParams,
                                                      float inverseMass)
{
    switch (shapeType)
    {
    case physics::ColliderShapeType::Sphere:
        return computeSphereInverseInertia(shapeParams.x, inverseMass);
    case physics::ColliderShapeType::Box:
        return computeBoxInverseInertia({shapeParams.x, shapeParams.y, shapeParams.z},
                                        inverseMass);
    case physics::ColliderShapeType::Capsule:
        return computeApproximateCapsuleInverseInertia(shapeParams.x, shapeParams.y,
                                                       inverseMass);
    }

    return {0.0f, 0.0f, 0.0f};
}

} // namespace cressim::neo::examples::helpers

#endif
