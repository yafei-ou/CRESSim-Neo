#include "physics/soft_body_utilities.h"

#include "common/math_utils_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace cressim::neo::physics
{

namespace
{

struct Int3
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const Int3 &rhs) const noexcept
    {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct Int3Hasher
{
    std::size_t operator()(const Int3 &value) const noexcept
    {
        std::size_t seed = static_cast<std::size_t>(value.x) * 73856093u;
        seed ^= static_cast<std::size_t>(value.y) * 19349663u;
        seed ^= static_cast<std::size_t>(value.z) * 83492791u;
        return seed;
    }
};

Int3 gridCoord(const Diligent::float3 &position, float cellSize)
{
    const float invCellSize = cellSize > 0.0f ? 1.0f / cellSize : 1.0f;
    return Int3{static_cast<int>(std::floor(position.x * invCellSize)),
                static_cast<int>(std::floor(position.y * invCellSize)),
                static_cast<int>(std::floor(position.z * invCellSize))};
}

Diligent::float3 rotateVector(const Diligent::QuaternionF &rotation, const Diligent::float3 &value)
{
    return rotation.RotateVector(value);
}

Diligent::QuaternionF inverseQuaternion(const Diligent::QuaternionF &rotation)
{
    return Diligent::QuaternionF{-rotation.q.x, -rotation.q.y, -rotation.q.z, rotation.q.w};
}

Diligent::float3 boxClosestPointLocal(const Diligent::float3 &point,
                                      const Diligent::float3 &halfExtents)
{
    return Diligent::float3{std::max(-halfExtents.x, std::min(point.x, halfExtents.x)),
                            std::max(-halfExtents.y, std::min(point.y, halfExtents.y)),
                            std::max(-halfExtents.z, std::min(point.z, halfExtents.z))};
}

} // namespace

std::vector<SoftRigidBroadPhaseCandidate> buildSoftRigidBroadPhaseCandidatesCpu(
    const SoftParticleSoAHost &softParticles, const RigidSurfaceParticleSoAHost &surfaceParticles,
    float cellSize)
{
    std::vector<SoftRigidBroadPhaseCandidate> candidates;
    if (softParticles.empty() || surfaceParticles.empty())
    {
        return candidates;
    }

    const float safeCellSize = std::max(cellSize, 1.0e-4f);
    std::unordered_map<Int3, std::vector<std::uint32_t>, Int3Hasher> grid;
    grid.reserve(surfaceParticles.size());

    for (std::uint32_t surfaceIndex = 0u; surfaceIndex < surfaceParticles.size(); ++surfaceIndex)
    {
        grid[gridCoord(Diligent::float3{surfaceParticles.worldPositions[surfaceIndex].x,
                                        surfaceParticles.worldPositions[surfaceIndex].y,
                                        surfaceParticles.worldPositions[surfaceIndex].z},
                       safeCellSize)]
            .push_back(surfaceIndex);
    }

    for (std::uint32_t softIndex = 0u; softIndex < softParticles.size(); ++softIndex)
    {
        const Diligent::float3 softPosition{softParticles.positionsInvMass[softIndex].x,
                                            softParticles.positionsInvMass[softIndex].y,
                                            softParticles.positionsInvMass[softIndex].z};
        const Int3 baseCoord = gridCoord(softPosition, safeCellSize);
        std::unordered_set<std::uint32_t> emittedBodies;

        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const Int3 coord{baseCoord.x + dx, baseCoord.y + dy, baseCoord.z + dz};
                    const auto it = grid.find(coord);
                    if (it == grid.end())
                    {
                        continue;
                    }

                    for (const std::uint32_t surfaceIndex : it->second)
                    {
                        if (surfaceParticles.environmentIndices[surfaceIndex] !=
                            softParticles.environmentIndices[softIndex])
                        {
                            continue;
                        }
                        if ((softParticles.collisionMasks[softIndex] &
                             surfaceParticles.collisionLayers[surfaceIndex]) == 0u ||
                            (surfaceParticles.collisionMasks[surfaceIndex] &
                             softParticles.collisionLayers[softIndex]) == 0u)
                        {
                            continue;
                        }

                        const Diligent::float3 delta{
                            surfaceParticles.worldPositions[surfaceIndex].x - softPosition.x,
                            surfaceParticles.worldPositions[surfaceIndex].y - softPosition.y,
                            surfaceParticles.worldPositions[surfaceIndex].z - softPosition.z};
                        const float radius = softParticles.radii[softIndex] +
                                             surfaceParticles.sampleRadii[surfaceIndex];
                        if (Diligent::dot(delta, delta) > radius * radius)
                        {
                            continue;
                        }

                        if (emittedBodies
                                .insert(surfaceParticles.owningRigidBodyIndices[surfaceIndex])
                                .second)
                        {
                            candidates.push_back(
                                {softIndex, surfaceParticles.owningRigidBodyIndices[surfaceIndex]});
                        }
                    }
                }
            }
        }
    }

    return candidates;
}

bool computeSoftRigidContactCpu(const SoftParticleSoAHost &softParticles,
                                const std::uint32_t softParticleIndex,
                                const RigidBodyState &rigidBody, std::uint32_t rigidBodyIndex,
                                const ColliderState &collider, std::uint32_t colliderIndex,
                                SoftRigidContact &outContact)
{
    if (softParticleIndex >= softParticles.size())
    {
        return false;
    }

    if (softParticles.environmentIndices[softParticleIndex] != collider.environmentIndex)
    {
        return false;
    }
    if ((softParticles.collisionMasks[softParticleIndex] & collider.collisionLayer) == 0u ||
        (collider.collisionMask & softParticles.collisionLayers[softParticleIndex]) == 0u)
    {
        return false;
    }

    const Diligent::float3 worldColliderPosition =
        rigidBody.position + rotateVector(rigidBody.rotation, collider.localPosition);
    const Diligent::QuaternionF worldColliderRotation = rigidBody.rotation * collider.localRotation;
    const Diligent::QuaternionF invRotation           = inverseQuaternion(worldColliderRotation);
    const Diligent::float3 particleLocal              = rotateVector(
        invRotation, Diligent::float3{softParticles.positionsInvMass[softParticleIndex].x,
                                      softParticles.positionsInvMass[softParticleIndex].y,
                                      softParticles.positionsInvMass[softParticleIndex].z} -
                         worldColliderPosition);

    Diligent::float3 normalLocal{0.0f, 1.0f, 0.0f};
    float signedDistance = 0.0f;

    switch (collider.shapeType)
    {
    case ColliderShapeType::Sphere:
    {
        const float radius = std::max(collider.shapeParams.x * rigidBody.scale.x, 1.0e-4f);
        const float length = std::sqrt(Diligent::dot(particleLocal, particleLocal));
        normalLocal =
            length > 1.0e-5f ? particleLocal / length : Diligent::float3{0.0f, 1.0f, 0.0f};
        signedDistance = length - radius;
        break;
    }
    case ColliderShapeType::Box:
    {
        const Diligent::float3 halfExtents{collider.shapeParams.x * rigidBody.scale.x,
                                           collider.shapeParams.y * rigidBody.scale.y,
                                           collider.shapeParams.z * rigidBody.scale.z};
        const Diligent::float3 closest = boxClosestPointLocal(particleLocal, halfExtents);
        const Diligent::float3 delta   = particleLocal - closest;
        const float deltaLength        = std::sqrt(Diligent::dot(delta, delta));
        if (deltaLength > 1.0e-5f)
        {
            normalLocal    = delta / deltaLength;
            signedDistance = deltaLength;
        }
        else
        {
            const Diligent::float3 distances{halfExtents.x - std::abs(particleLocal.x),
                                             halfExtents.y - std::abs(particleLocal.y),
                                             halfExtents.z - std::abs(particleLocal.z)};
            if (distances.x <= distances.y && distances.x <= distances.z)
            {
                normalLocal = Diligent::float3{particleLocal.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f};
                signedDistance = -distances.x;
            }
            else if (distances.y <= distances.z)
            {
                normalLocal = Diligent::float3{0.0f, particleLocal.y >= 0.0f ? 1.0f : -1.0f, 0.0f};
                signedDistance = -distances.y;
            }
            else
            {
                normalLocal = Diligent::float3{0.0f, 0.0f, particleLocal.z >= 0.0f ? 1.0f : -1.0f};
                signedDistance = -distances.z;
            }
        }
        break;
    }
    case ColliderShapeType::Capsule:
    {
        const float radius = std::max(
            collider.shapeParams.x * std::max(rigidBody.scale.x, rigidBody.scale.z), 1.0e-4f);
        const float halfHeight = std::max(collider.shapeParams.y * rigidBody.scale.y, 0.0f);
        const Diligent::float3 segmentA{0.0f, -halfHeight, 0.0f};
        const Diligent::float3 segmentB{0.0f, halfHeight, 0.0f};
        const Diligent::float3 ab = segmentB - segmentA;
        const float abLengthSq    = Diligent::dot(ab, ab);
        float t                   = 0.0f;
        if (abLengthSq > 1.0e-6f)
        {
            t = std::max(0.0f,
                         std::min(1.0f, Diligent::dot(particleLocal - segmentA, ab) / abLengthSq));
        }
        const Diligent::float3 closest = segmentA + ab * t;
        const Diligent::float3 delta   = particleLocal - closest;
        const float length             = std::sqrt(Diligent::dot(delta, delta));
        normalLocal    = length > 1.0e-5f ? delta / length : Diligent::float3{1.0f, 0.0f, 0.0f};
        signedDistance = length - radius;
        break;
    }
    }

    const float penetration = softParticles.radii[softParticleIndex] - signedDistance;
    if (penetration <= 0.0f)
    {
        return false;
    }

    outContact.softParticleIndex = softParticleIndex;
    outContact.rigidBodyIndex    = rigidBodyIndex;
    outContact.colliderIndex     = colliderIndex;
    outContact.normal            = common::runtime_math::safeNormalize(
        rotateVector(worldColliderRotation, normalLocal), Diligent::float3{0.0f, 1.0f, 0.0f});
    outContact.penetration = penetration;
    return true;
}

} // namespace cressim::neo::physics
